//! Kalman / random-walk-beta dynamic linear model (docs/PLAN.md §4.4).
//!
//! State per target: coefficient mean `b_j` and covariance `P_j` (k x k).
//! Observation `y_j = z' b_j + e`, `e ~ N(0, R_j)`; coefficients follow a random
//! walk `b_j <- b_j + w`, `w ~ N(0, Q)`.
//!
//! Per row (clock delta `d`, weight `w_row`):
//!
//! ```text
//! b_j <- Phi b_j                      (transition; Phi = diag(2^(-d/r_i)))
//! P_j <- Phi P_j Phi                  (every row)
//! D_j <- D_j + d                      (the clock since P_j last took an observation)
//! P_j <- P_j + Q * D_j^2, D_j <- 0    (only on a row that observes P_j's target;
//!                                      on a reverting slot D^2 is g_i(D)^2, below)
//! s    = z' P_j z + R_j / w_row       (innovation variance)
//! k    = P_j z / s                    (gain)
//! b_j <- b_j + k (y_j - z' b_j)
//! P_j <- P_j - k z' P_j
//! ```
//!
//! with `P_j` and `Q` in units of `R_j`, `R_j` read as 1 in `s`, wherever the
//! noise is learned (below).
//!
//! The last line is the covariance form, the default ([`CovarianceForm`],
//! a spec's `covariance_form`). `P_j` may instead be held as its square
//! root, a triangular `C` with `P_j = C C'`, updated by Carlson's step
//! (`kalman/forms.rs` states it): it rounds at `u sqrt(p0)` where `P_j`
//! rounds at `u p0`, so a huge prior costs it no digits, at O(k³) a row
//! under process noise for O(k²).
//!
//! **The noise is a scale, not a weight (docs/PLAN.md task 278).** The
//! model's noise `R_j` is one constant per target, and the prior `p0 R_j` and
//! the drift `R_j (ln 2 / h)²` are multiples of it, so `P_j / R_j` and the
//! gain `P z / (z' P z + R / w)` never read it: the conjugate form of a
//! regression with an unknown noise variance (West and Harrison, *Bayesian
//! Forecasting and Dynamic Models*, §4.5). So without `obs_var` (and under
//! [`KalmanCfg::noise_units`], which every model the bank builds sets) `P_j`
//! is held in units of the noise: the gain is `P z / (z' P z + 1 / w)`, the
//! derived process noise `(ln 2 D / h)²`, and the noise estimate, the EW
//! residual variance, only scales what is read out -- `pred_var`,
//! `se_coef`, the readiness gate, which is `z' P⁻ z` already in units of the
//! noise -- and measures an explicit `q`, `q / R_j`, the one place it reaches
//! the gain. An estimate inside the gain weighs each row by the inverse of
//! the noise as it stood before the row, a weighted least squares whose
//! weights a burst of large residuals sets: the burst's rows weigh as before
//! it, and the clean rows after it count for less. With no process noise the
//! burst never washes out. Task 278 (a) measured, seed 31, 2,000 rows, 20%
//! of the first 500 rows 50 too high where `x0 > 0`: slopes [-25.5, 33.6]
//! against least squares' [4.1, -1.4] on the same rows; after a change of
//! regime the fit stayed 45.7 of a 46 gap away; Cauchy noise in the first
//! rows left it 658 off and growing. In units of the noise each is Bayesian
//! regression on the same rows: within 2.4e-15 of faer's solve of the
//! posterior (`kalman/tests/noise_units.rs`). The same weighting is
//! volatility weighting on returns whose volatility clusters, and
//! docs/VALIDATION.md's intraday returns had scored R² -0.059 and -0.075 with
//! it, -0.070 and -0.101 without; a fixed `obs_var` of any size scores as
//! the learned noise does now (-0.06984 and -0.10019), which is the
//! cancellation measured. A step in the truth inflates the residual
//! variance too, and with it in the gain a step was learned about 1.45 times
//! slower than with the true noise.
//!
//! **The intercept's prior is diffuse.** In units of the noise the prior
//! needs no noise: `p0` on each slope -- from the start unstandardized, on
//! the first observed row its scale is usable standardized -- and on the
//! intercept no prior at all. The first row that observes a covariance's
//! target takes the exact step of a prior diffuse in the intercept (Durbin
//! and Koopman, *Time Series Analysis by State Space Methods*, §5.2): `b_0`
//! takes the whole innovation, `b_0 += y − z' b`, `P_00 = z' P z + 1 / w`
//! and `P_0i = −(P z)_i`, the slopes' block unmoved -- the limit of the
//! ordinary update as the intercept's prior variance grows without bound
//! ([`diffuse_step`]). Until then the intercept's row and column are 0 and
//! nothing writes them. A target first seen after its shared covariance
//! took that row has its intercept centred the same way, `b_0` taking its
//! first innovation. So a level `L` in the target moves every prediction by
//! `L`, to the rounding of a number of `L`'s size, from the first: within 17
//! `u (|L| + max|y|)` at 1e4, -1e6 and 1e8. When the first innovations sized
//! the noise and the prior, the level was among them (task 278 (b)'s K1):
//! at `coef_half_life` 200 and a half-life of 300, a level of 1e4 left the
//! predictions 0.019 to 0.032 of the target's spread off over rows 3,000 to
//! 6,000, for about `log₂(L² / σ²)` half-lives. With no process noise and
//! the noise learned, the filter is least squares with the ridge `1 / p0`
//! on each slope and none on the intercept. The costs: an absurd target, at
//! the input bound, now moves the coefficients as any observation does, and
//! the drift forgets it at `coef_half_life`, where the inflated estimate
//! inflated the derived process noise and `P` forgot everything within a
//! few rows; and under a half-life the filter weighs its first rows evenly
//! until `P` meets the drift, some `h / ln 2` rows, which keeps a burst
//! there about 2.7 times longer than EW least squares' weights do.
//!
//! A state saved before task 278 has no `noise_units` and goes on as it
//! was fitted, `P` absolute and the estimate in the gain, every rule the
//! sections below give for it; with `obs_var`, `P` is absolute either way
//! and nothing here changes. The sections below say which form each rule
//! is.
//!
//! **Reversion (ENHANCEMENTS E41).** `revert_half_life` gives each slot a
//! reversion half-life `r_i`: between observations the coefficient mean
//! shrinks toward zero by `2^(-d/r_i)`, so a coefficient no row has
//! supported for a while is forgotten rather than carried. `r_i = inf` (the
//! default) is `Phi = I`, the random walk, and costs nothing. The pull is
//! toward zero in the *standardized* coordinates when `standardize` is on:
//! a slope toward "no effect", the intercept toward "the target averages
//! zero"; give the intercept `inf` to leave it a random walk. The transition
//! runs on every row, the ones that observe nothing included: `Phi` is
//! multiplicative over a split gap, `2^(-d₁/r) 2^(-d₂/r) = 2^(-(d₁+d₂)/r)`,
//! so where a gap is cut does not move it.
//!
//! **A reverting slot's process noise is bounded in the gap (docs/PLAN.md
//! task 214).** For an observation `D` after the last it is `q_i g_i(D)²`,
//!
//! ```text
//! g_i(D) = (1 - 2^(-D/r_i)) / theta_i,    theta_i = ln2 / r_i      (g_i(D) = D at r_i = inf)
//! ```
//!
//! the displacement of a velocity of variance `q_i` held over the gap while
//! the reversion pulls, `db = (u - theta_i b) dt`, which `q_i D²` is without
//! the pull. It is `q_i D²` for a gap well under `r_i` (0.993 of it at `D =
//! r_i/100`, 0.933 at `r_i/10`), so the matching below holds there, and it
//! saturates at `q_i / theta_i²` past `r_i` (0.9998 of it at `20 r_i`):
//! across a run of rows that observe nothing the coefficient's uncertainty is
//! bounded, as a mean-reverting coefficient's is. Charged `q_i D²` for the
//! whole gap (task 211) it grew without bound; charged per row, as before
//! that, it settled at `q_i / (1 - phi_i²)`, a number of the rows the gap was
//! cut into. At observations one clock unit apart the noise is `q_i g_i(1)²`,
//! short of `q_i` by about `ln2 / r_i`: 10.8% at `r_i` = 6, 2.7% at 25, 0.7%
//! at 100, which moved a prediction by at most 0.26, 0.03 and 0.004 noise
//! deviations on a dense stream whose slope drifts. With `Q` from `coef_half_life` a reverting
//! slot settles, at observations `D` apart, at the prior variance `q_i
//! g_i(D)² / (1 - phi_i²)`, a stationary AR(1) instead of an unbounded walk
//! (for `D` well under `r_i` that is about `q_i D r_i / (2 ln2)`, which grows
//! with the spacing as the gain matching's own variance does).
//!
//! **Process noise from a per-factor half-life.** On standardized features, the
//! steady-state gain of a random-walk-beta filter matches EW-RLS with half-life
//! `h_i` when the noise added for an observation `D` clock units after the
//! last is `sigma^2 * (ln2 * D / h_i)^2 = q_i D^2` with `q_i = sigma^2 *
//! (ln2 / h_i)^2` (docs/PLAN.md §4.4, task 150): EW-RLS forgets `2^(-D/h)` of
//! its information between the two observations, a half-life of `h/D`
//! observations, and that is the half-life the matching is done at. Per
//! direction, with `p = P / R` before the observation, the filter's steady
//! state solves `p² / (p + 1) = c²`, `c = ln2 D / h`, and its gain `p / (p +
//! 1)` is EW-RLS's `1 - 2^(-D/h)` to first order in `D/h`: the exact noise
//! is `sigma^2 g^2 / (1 - g)` with `g = 1 - 2^(-D/h)`, which `c²` is within
//! 1% of for observations closer than half a half-life and 4% at one.
//! `half_life` may be scalar or per factor; `half_life = inf` gives `q_i =
//! 0`, pinning that coefficient. An explicit `q` overrides the derivation
//! and is added as `q_i D^2` too: the noise an observation one clock unit
//! after the last adds, not a variance per unit of elapsed clock.
//!
//! **The noise is charged once per observation, for the whole clock since
//! the last (docs/PLAN.md task 211).** `D` is the clock since the
//! covariance last took a row that observes its target -- present, at a
//! positive weight; under `share_p` any of the targets -- and a row that
//! observes none, a null target or a weight of 0, only adds its `d` to it.
//! The noise is not additive over a split gap: `q (d₁² + d₂²) < q (d₁ +
//! d₂)²`. Charged per row, as it was, a row of weight 0 inside a gap -- which
//! must advance the clock and teach nothing (hard rule 9) -- shrank the
//! gap's noise and moved every later prediction (0.197 at `coef_half_life`
//! 20 on the review's stream of 2026-10-08), and so did a null target, and
//! `rp.stream.embargo`'s doubled stream, whose predict rows are such rows,
//! parted from the native embargo. And a target seen on one row in `n` took
//! `n` charges of `q d²` where one of `q (n d)²` was due: its memory grew as
//! `h √n`, 166, 270 and 462 clock units at `n` = 4, 10 and 25 against 74.5
//! seen on every row. Charged per observation, the matching above is made at
//! the observations' own spacing `D`, whatever rows lie between them, so
//! `coef_half_life` is a half-life on the clock for a sparse target as for
//! a dense one: EW-RLS's `2^(-D/h)` is the same over one gap of `D` as over
//! any cut of it. The transition stays per row, being multiplicative. A
//! `gap_cap` caps each row's `d` before it reaches the model, so a gap cut
//! into rows escapes a cap that the gap whole would meet: the doubled
//! stream's sub-gaps do, for every model, and it is an oracle for the
//! native embargo only without a cap.
//!
//! **What `Q` means in the caller's units.** Under `standardize` the noise
//! is defined per standard deviation of each feature, as the moments stand
//! when it is charged: a slope's random walk in the caller's units has a
//! variance per observation of `q D² / s²`, a drift measured in the feature's
//! current spread. A feature whose spread doubles has its slope's walk
//! slowed by four in its own units; measured on the review's stream of
//! 2026-10-08, a slope's `se_coef²` grew by `sigma² (ln2 / h)² / s²` a
//! unit of clock at `s` = 1 and 10 alike (ratio 1.015), when the noise was
//! charged per row.
//!
//! **Memory under correlated features.** `coef_half_life` is the memory of
//! each standardized direction with the design's other directions held
//! still. Where features are correlated, the information along an
//! eigendirection of the design's correlation with eigenvalue `λ` is `λ`
//! per row, and a random walk of the same variance in every direction is
//! learned along it with a memory of `h / √λ` (Ljung and Gunnarsson 1990):
//! measured at a correlation of 0.9 and `h` = 50, 37.5 and 160.5 clock units
//! along `[1, 1]` (`λ` = 1.9) and `[1, -1]` (`λ` = 0.1), against `h/√λ` =
//! 36.3 and 158.1 (with `obs_var` the true noise). EW-RLS forgets every
//! direction at `h` alike (51.5 and 52.5 there), so the two match at any
//! spacing only on uncorrelated features. And with `R` the EW residual
//! variance, which a step in the truth inflates before `P` catches up, the
//! step response is slower than with the true noise: 72.5-74.5 clock units
//! against 50-51.5 at `h` = 50 on uncorrelated features, about 1.45 times
//! (with the noise estimate in the gain; in units of the noise the gain
//! reads none, below).
//!
//! **Standardized coordinates, held at an anchor (docs/PLAN.md task 211;
//! task 206 before it).** Features are standardized against a shared
//! [`EwDiag`] over `z`, EW means and variances, O(k) a row: `z_0 = 1` and
//! `z_i = (x_i − m_i) / s_i` (`x_i / s_i`, the root of the raw second
//! moment, without an intercept; review 2026-09-12, C10), against the
//! moments before the row. `R_j` defaults to the EW residual variance
//! `sigma^2_j` unless `obs_var` is given. `b` and `P` are held in the
//! coordinates of an *anchor*: the moments' means, as pairs, and scales at
//! the last re-map, `m^a` and `s^a`. The row is read in them, `z^a_i = (x_i −
//! m^a_i) / s^a_i`, and so is the prediction `z^a' b`. With `m`, `s` the
//! moments as they stand, `z^a = M z` for every row, with
//!
//! ```text
//! M_00 = 1,    M_i0 = c_i = (m_i − m^a_i) / s^a_i   (0 without an intercept),    M_ii = a_i = s_i / s^a_i
//! ```
//!
//! so `b` is `b_cur = Mᵀ b` in the current coordinates and `P` is `Mᵀ P M`:
//! no prediction and no predictive variance moves because the moments did,
//! and the coefficients in the caller's units move only with the fit. The
//! reversion, the process noise and the prior are defined in the current
//! coordinates, so a half-life and `p0` mean the same thing whatever a
//! column's units, and carried into the anchor's exactly. The noise `Q D²`,
//! diagonal in the current coordinates, is `M⁻ᵀ Q M⁻¹` in the anchor's, an
//! arrowhead -- the diagonal plus the intercept's row and column -- and O(k)
//! a row:
//!
//! ```text
//! v_i = c_i / a_i:    P_00 += Σ_i v_i² q_i D²,    P_0i = P_i0 += −v_i q_i D² / a_i,    P_ii += q_i D² / a_i²
//! ```
//!
//! A slot's prior `p0 R` is carried in the same way, and the transition `Phi`
//! becomes `M⁻ᵀ Phi Mᵀ`: `b_0 <- phi_0 b_0 + Σ_i g_i b_i` with `g_i = c_i
//! (phi_0 − phi_i)`, `b_i <- phi_i b_i`, which is `Phi` itself where every
//! slot reverts at one half-life. Every readout -- the prediction, `pred_var`,
//! the coefficients and their variances, the readiness statistic -- reads
//! the anchor's coordinates or maps to the current ones exactly.
//!
//! **The re-map.** When the moments drift from the anchor -- a scale by
//! more than a factor of `1 + τ` either way, `max(a_i, 1/a_i) − 1 > τ`, or a
//! mean by more than `τ` of the anchor's scale, `|c_i| > τ`, over the slots
//! that hold anything, with `τ` = 1 (`ANCHOR_DRIFT`) -- `b` and `P` are
//! mapped to the moments as they stand after the row, `b <- Mᵀ b` and `P <-
//! Mᵀ P M`:
//!
//! ```text
//! b'_0 = b_0 + Σ_i c_i b_i,    b'_i = a_i b_i
//! u = P_0· + Σ_l c_l P_l·,   P'_00 = u_0 + Σ_j c_j u_j,   P'_0j = P'_j0 = a_j u_j,   P'_ij = (a_i a_j) P_ij
//! ```
//!
//! and the anchor becomes those moments. The drift bounds the arrowhead's
//! terms, `v_i² ≤ 4`, so no row reads a large number back through a
//! cancellation. Measured on the review's regimes (30,000 rows, `k` 1 and
//! 5, R² 0.978 and 0.99998, half-lives 10 to none) against the per-row
//! re-map of task 206 run on the same rules: within 1.3e-12 of a noise
//! standard deviation at `coef_half_life` 50, and 2.0e-11 with no process
//! noise, where two per-row implementations of it (Rust and numpy) differ
//! by 4.7e-12; 3 to 5 re-maps in 30,000 rows of stationary data, most of
//! them in the first rows, and 20 to 45 at a half-life of 10. A drift in
//! `|a_i − 1|` alone never re-maps a shrinking scale, and leaves `a_i` tiny
//! with arrowhead terms of `q / a_i²` that cancel in every prediction: on a
//! feature whose spread shrinks 3% a row the anchored filter with that
//! drift parted from the per-row re-map by 8e9 noise standard deviations,
//! where this one keeps to 2.7e-14. Task 206 re-mapped on every row, four
//! sweeps over `P`: at `k = 10` a row cost 277 ns, and costs 221 here
//! (2026-10-08, median of nine interleaved runs at a load of 5 to 7).
//!
//! A row of weight 0 moves no moment and nothing is re-mapped. A re-map past
//! what a stretch of data makes -- a scale by more than a factor of 1024
//! either way, or a mean by more than 1024 of the anchor's scale -- is
//! refused whole, and so is one any of whose numbers would not be finite:
//! `b` and `P` keep their numbers and are read in the new coordinates, as
//! the row's other updates that would not be finite are skipped
//! (docs/IMPROVEMENTS.md C2). The re-map puts `c² P_ii`
//! into the intercept's variance, and every later row reads it back through
//! a cancellation, so a shift of `c` costs `c²` rounding steps of `P`'s
//! narrowest direction: at the input bound, where a feature of 1e100 shifts
//! a mean by about 1e99 of its scale, the re-map left a slope's variance at
//! −3.6e149 and the reverting filter never recovered; at 1024 the cost is
//! 2.3e-10 of it.
//!
//! **After a long gap** the moments' history is a share `a` of the moments
//! with the row in, and the row alone would move them by a scale of
//! `sqrt(a)` of the feature's spread, past what the re-map carries (after
//! 30 half-lives, an error 3e9 times the same rows' with no gap, at a
//! half-life of 20 rows). So on a row more than one half-life after the
//! moments last moved, the moments read their history at no less than
//! the row's own weight, up to the weight the history had, `H = max(lam·W,
//! min(W, w))`: shares `H/(H + w)` and `w/(H + w)`. (Uncapped, a history
//! of weight `1e-6` read against a row of weight 1 took half the share.)
//! The weight kept is `lam·W + w` as ever, so `n_eff` is the
//! plain recursion's, and a row within a half-life of the last one that
//! moved the moments is the plain update, to the bit. The gap is the
//! moments' own: the clock since the last row of weight above 0, which a
//! zero-weight row adds to and does not clear. A row that observes no
//! target still moves the moments, so it closes the gap; read from the
//! clock since a target was observed, the floor opened on rows whose
//! moments had moved one row before (16 of 20 such rows at a half-life of
//! 5).
//!
//! **The first rows (docs/PLAN.md task 211).** In units of the noise each
//! slope's prior is `p0` and the intercept's diffuse (above); the rest of
//! this section is the form before, and `obs_var`'s. Over the first rows the
//! moments rest on nothing: on row 0 a feature is read against a mean of 0
//! and a scale of 1, raw, and on row 1 against a variance of 0. So each
//! coefficient's prior is set on the first row that observes the target on
//! which its feature's scale is usable -- a variance above 0 (with an
//! intercept), a raw second moment above 0 (without) -- and until then the
//! coefficient is 0 and its variance 0: it takes no correction and no
//! process noise, and a row's raw reading of the feature reaches nothing.
//! A target none of whose slots is sized is not predicted: its prediction
//! would be the uncorrected coefficients', `0.0` whatever its level, and
//! would seed the residual variance with the level squared (docs/PLAN.md
//! task 270, review round 9 B-1).
//! The intercept's scale is always usable. A feature whose variance becomes
//! usable later than the others' gets its own prior then. Under
//! `standardize` the prior of slot `i` is `P_ii = p0 R`, carried into the
//! anchor's coordinates, with `R` the weighted median of the first three
//! rows' squared innovations that observed the target (each `e = y − z' b`
//! before its row's update; a row counts once, at its mean square over the
//! targets it observes, at its weight), over 0.4549, the median of a χ²₁:
//! what a median of Gaussian innovations' squares reads of their variance,
//! the noise `p0` is defined against. It is set once three rows give it
//! (`PRIOR_ROWS`) and fixed from then on, so a slot sized later reads the
//! same `R`. With `obs_var`, `R = obs_var`, and the intercept's prior is set
//! from the start. Three is the number of rows a feature's first usable
//! scale rests on in a stream with a target on every row (two for the
//! moments' variance, and the row itself), so there the intercept and the
//! features are sized together. With no process noise a prior sized too
//! small pins the fit for good, and one squared innovation did: the
//! review's seed whose sizing innovation was 0.004 on a target of spread 4.5
//! read slopes of 0.3 to 1.2 against a truth of 2 after 30,000 rows. One
//! square falls below 1% of the noise with probability 0.08, the median of
//! three as `R` reads it with probability 0.0084, and their mean, which `R`
//! was until task 214, with 0.0014. **A median and not a mean** (docs/PLAN.md
//! task 214): a target at the input bound among the first rows, a square of
//! `1e200`, made the mean `3e199`; `P` began there, the update left a slot's
//! variance at `-1.6e184`, and the filter never learned again, its slopes at
//! `1.8e-101` and `1.20` after 30,000 rows against a truth of 1 and -0.5.
//! The median of three ignores one such row. And a covariance whose
//! diagonal leaves the numbers of at least 0 -- the sign of a lost
//! positive-definiteness a row can check in O(k) -- is sized again at the
//! end of the row, from the noise as it stands (`obs_var`, else the
//! residual variance), as at the start; with no noise yet it is left
//! unsized for the next row that observes its target to size.
//! The state follows the moments from the first row; the 22-row warm-up
//! `sgd` and `pa` keep ([`crate::Warmup`]) is for their step sizes, and the
//! filter's `P` carries its own uncertainty through the first rows. On task
//! 74's short histories (200-row groups, 20 features, half-life 50) the
//! warm-up and the prior from row 0 gave R² 0.722 on rows 25 to 50; this
//! gives 0.95, and a level of 1e8 in the features no longer moves any
//! prediction past the first rows (0.87 before). Features scaled by a power
//! of two predict the same to the bit: no row reads a raw `x`.
//!
//! **`min_weight` withholds the output alone (docs/PLAN.md task 278,
//! D10).** The residual variance `σ²` learns from the filter's own
//! prediction on every row it predicts, whether or not `min_weight` lets
//! that prediction out, so the fit -- `b`, `P` and `σ²` -- is the same at
//! any `min_weight`. The gate held learning back too: a withheld row taught
//! `σ²` nothing, and an explicit `q`, which in units of the noise is read
//! over `σ²`, followed the rows the gate let through (a slope 0.0064 off
//! the ungated fit's at 2,000 rows, `q = 1e-6` on each slot and
//! `min_weight` 50, worst of 50 seeds). A state written before schema 64
//! goes on that way ([`KalmanCfg::min_weight_holds_learning`]).
//!
//! **The noise before the first residual (review 2026-10-05, CC4).** In
//! units of the noise only an explicit `q` reads the noise, and this rule
//! for it; the gain reads none. A target has no residual variance until a row with a prediction for it
//! gives it a residual, and a prediction needs an earlier row of the target
//! and its covariance sized. Until then a row's noise is its own innovation
//! squared, `R_j = e_j^2` with `e_j = y_j - z' b_j`, computed before the
//! update and so out of sample, as the prediction is. The `sigma^2` the
//! process noise is derived from is the same number. On the row that gives
//! `sigma^2_j` its first residual, `e_j` is that residual, so the noise
//! starts where the residual variance will. The noise was the literal 1, in
//! the target's units, and the gains of a warm-up depended on those units.
//! A `sigma^2_j` of 0 (every residual so far exactly 0) counts as none.
//!
//! An innovation of exactly 0, or one whose square is not finite, sizes no
//! noise. At `R = 0` the update would take the row as exact and collapse
//! `P` along `z`, so such a row corrects nothing. Neither does a row with
//! no innovation, a null target or a weight of 0. Under `share_p` the noise
//! is the mean `sigma^2` over the targets that have one, and before any has
//! one, the mean of the squared innovations of the targets the row observes.
//! A given `obs_var` is the noise throughout. [`Kalman::pred_var`] keeps NaN
//! for `R_j` until there is a residual variance: it describes a prediction,
//! which is made before the row's target is seen.
//!
//! **The prior without `standardize` is `p0` times the first noise (CC4).**
//! In units of the noise it is `p0` on each slope from the start, the
//! intercept diffuse (above). In the form before, `P_j` starts unsized, all
//! zero. The first row with a noise sets it to
//! `P_0 = p0 R I`, with `R` that row's noise, and the row's gain reads it;
//! that row adds no process noise. So `p0` is a ratio: at 1 the prior is as
//! uncertain as one observation, the conjugate prior of a regression with an
//! unknown noise variance. With `obs_var` given, `P_0 = p0 obs_var I` from
//! the start, and with no process noise the filter is the ridge regression
//! with penalty `1 / p0` -- unstandardized only: standardized, each slot's
//! prior waits for its feature's scale, as above. Scaling every target by
//! `c` (with any `obs_var` or explicit `q` by `c^2`) scales every
//! prediction by `c`, at any `p0`. A slot whose variance has decayed to
//! exactly 0 through the reversion holds nothing, and is sized again.
//!
//! **One covariance for every target (`share_p`).** In units of the noise
//! `P`'s recursion reads no noise, so targets observed on the same rows carry
//! the same `P`, and the shared filter predicts as the per-target ones do, to
//! the bit; it parts from them only where a target is null on rows another
//! is present on, whose rows the shared `P` takes too, and an explicit `q` is
//! measured against the mean noise. Target `j`'s covariance is `σ²_j P`. In
//! the form before, `P` is per target because the Riccati recursion depends
//! on `R_j`. With `share_p` the filter
//! keeps one `P` driven by the mean `sigma^2` over the targets that have one
//! (docs/PLAN.md §4.4 [validate]), as it stands when the row arrives, summed
//! in ascending order. A target with no residual variance yet has no
//! estimate of the noise, not a noise of 0: counted in the mean, a target
//! null for its first rows halved the noise the others were weighed against,
//! and moved their predictions (review round 5, A1). `P`'s recursion reads
//! `z`, `R` and the weight, never `y`, so targets that share `R` would each
//! carry the same `P`: the shared one is that `P`. Every target observed on
//! the row takes its gain from `P` as the row finds it, and `P` takes the
//! row once, after them, if any target updated. So the order of the targets
//! moves no bit of any prediction, and a target beside an exact copy of
//! itself present on the same rows predicts as it would alone, to the bit. A
//! copy null on rows its twin is present on is not that: until it has a
//! residual variance it moves nothing, and from then its variance, learned
//! from fewer rows, enters the mean. Updating the shared `P` once per
//! target, as it did, counted each row once per target, as if the targets
//! shared their coefficients (docs/PLAN.md task 204).
//!
//! The shared `P` is the covariance of a filter whose noise is the mean
//! `σ̄²`. With a constant noise `σ²_j` per target, target `j`'s own filter
//! would keep `P_j = σ²_j P̃` and the shared one keeps `σ̄² P̃`, so target `j`'s
//! covariance is `P σ²_j / σ̄²`, and that is what its `se_coef` and
//! `pred_var` read (task 211). Read off `P` as it stands, two targets of
//! noise 0.01 and 1 had standard errors 7.3 times too large and 1.39 times
//! too small. A target with no residual variance yet is read at the mean.
//! The readiness statistic, `z' P_j z / σ²_j`, is `z' P z / σ̄²` for every
//! target, and needs no scaling.
//!
//! **When sharing `P` helps, and why the default does not share
//! (docs/PLAN.md task 214).** With a constant noise, `P/σ²` and the gain
//! `P z / (z' P z + σ²/w)` never read `σ²`: the process noise is `σ²` times
//! `(ln 2/h)²`, so the Riccati recursion of `P/σ²` is the same at any noise.
//! The shared filter and the per-target ones part only through how the
//! `σ²_j` paths move. So sharing helps where the targets are related, and
//! their noises move together; where each target's own `σ²_j` is a noisy
//! estimate -- a short history, a sparse target -- which the mean over the
//! targets smooths; and for cost, one `k×k` update a row for every target
//! where there were `m`. On docs/VALIDATION.md §4's two targets, the next
//! minute's return and the next five minutes', the shared filter scored an
//! R² of -0.0110 against -0.0168 on the first and -0.0880 against -0.0746 on
//! the second, in the form before; in units of the noise the two score the
//! same. The default stays one `P` per target.
//!
//! **Readiness (docs/PLAN.md task 116; docs/WARMUP-AND-CONVERGENCE.md
//! §2.1).** The filter knows the estimation variance of each prediction
//! under its own model: before row `z`'s target, at `d` clock units after the
//! last row, the prediction's variance over the noise is
//!
//! ```text
//! h(z) = z' P⁻ z / R,    P⁻ = Phi P Phi + Q (D + d)²
//! error_inflation = sqrt(1 + h(z))
//! ```
//!
//! with `z` standardized as the prediction is and `P⁻` the prior the row's
//! update starts from, after the transition and the process noise its clock
//! since the last observation adds. `R` is the noise as the state holds it:
//! `obs_var`, else the target's residual variance (the targets' mean under
//! `share_p`). Never the row's own innovation, which reads its target (hard
//! rule 2), so the ratio is infinite while `P` is unsized or the target has
//! no residual variance yet. `z' P⁻ z + R` is the prior predictive variance
//! a Kalman filter's `predict` step gives: exact where the model is
//! specified exactly, and per row, so the noise gate reads it per row.
//! `O(k²)`, paid only when the gate is set or the field emitted. It is
//! exact *under the model*: with `R` the EW residual variance of the
//! out-of-sample errors, which already carry the estimation error, the
//! ratio counts that error twice, and measured the mean squared error over
//! the mean `sigma² error_inflation²` is 0.96 at `coef_half_life` 50 (0.99
//! at 200).
//!
//! Without a row to read it at (the bank's summary) the statistic is its
//! mean field over the design, `sqrt(1 + Σ_i P_ii E[z_i²] / R)` at the
//! posterior, in the current coordinates: exact for uncorrelated features,
//! and above the average of `h` over the design while `P` tracks it (a
//! correlation matrix's inverse has a trace of at least `k`).
//!
//! Under process noise `P` never reaches 0. Per direction, on standardized
//! features with the half-life-derived `q`, `p = P⁻ / R` settles where
//! `p² / (p + 1) = c²`, `c = ln 2 · D / coef_half_life`, so
//! `p = (c² + sqrt(c⁴ + 4 c²)) / 2`, and the average row reads about
//! `sqrt(1 + k p)`: 1.07 at `k = 10`, `D = 1`, `coef_half_life = 50`.
//!
//! **Coefficient standard errors** (`se_coef`). `P` after the row is the
//! coefficients' posterior covariance in the anchor's coordinates, the noise
//! already in it; after a row that observes nothing, `P` as it stands
//! carries the noise of the clock since the last observation, `Q D²`, as the
//! next observation will charge it, so a coefficient's uncertainty grows
//! across a gap in its target (`pred_var` and the summary's readiness read
//! the same). Read out in the original units by the map `coefficients`
//! uses, `c_i = b_i / s^a_i` and `c_0 = b_0 − Σ_i c_i m^a_i`, it is `T P Tᵀ`,
//! whose diagonal [`OnlineModel::coef_variance`] reports. Exact under the
//! model, which says the coefficients walk: on a truth that does not move,
//! the process noise the half-life implies is variance the coefficients do
//! not have, and measured the standard errors are about √2 too large (a
//! mean squared error over `se²` of 0.47 to 0.50 at `coef_half_life` 50 and
//! 200, and 0.95 to 1.06 on a walk of the variance the half-life implies).
//!
//! **Per standard deviation, the readout before task 206.** Until task
//! 206 the state was read through the moments as they stood, which made the
//! model a filter on each feature's EW z-score: a coefficient per current
//! standard deviation, moving with the scale. That model is this one on a
//! z-scored feature column, which the window operators build in the stream
//! (docs/PLAN.md task 212): standardize the column there and give it to
//! `kalman` with `standardize=False`.

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, EwDiag};

mod forms;
mod horizons;
mod innovation;
mod nonlinear;
mod smooth;
pub use forms::CovarianceForm;
pub use nonlinear::{KalmanFamily, KalmanFilter, KalmanLink, KalmanObs, UkfNoise, UkfRule};
pub use smooth::{SMOOTH_MIN_WEIGHT, SMOOTH_STOOD, SMOOTH_UNSEEN, Smoothed, SmoothedRow};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct KalmanCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub fit_intercept: bool,
    /// Decay used for the standardization statistics and the EW residual
    /// variance (NOT for the coefficients: those follow the random walk).
    pub decay: Decay,
    /// Per-factor coefficient half-life in clock units (length 1 or `k_total`).
    /// `f64::INFINITY` pins a coefficient. Ignored when `q` is given.
    #[serde(with = "crate::humanfloat::vec_f64_or_tag")]
    pub half_life: Vec<f64>,
    /// Explicit process-noise variances (length `k_total`), overriding
    /// `half_life`.
    pub q: Option<Vec<f64>>,
    /// Fixed observation variance; defaults to the EW residual variance, and
    /// before a target has one, to the row's own innovation squared (the
    /// module doc, CC4).
    pub obs_var: Option<f64>,
    /// The prior variance of each coefficient, as a multiple of the first
    /// noise estimate: `P_0 = p0 R I`, set on the row that first sizes the
    /// noise, or from the start with `obs_var` (the module doc, CC4).
    pub p0: f64,
    pub share_p: bool,
    pub min_weight: f64,
    /// Per-slot reversion half-life in clock units (length 1 or `k_total`,
    /// intercept first): the coefficient mean shrinks toward zero by
    /// `2^(-d/r_i)` per row before the process noise is added. `f64::INFINITY`
    /// (the default) is the random walk. See the module doc.
    #[serde(default = "default_revert")]
    #[serde(with = "crate::humanfloat::vec_f64_or_tag")]
    pub revert_half_life: Vec<f64>,
    /// Standardize features internally before filtering (default), the
    /// filter's state following the moments once they have warmed up (the
    /// module doc; docs/PLAN.md task 206).
    ///
    /// On by default because the half-life-derived process noise
    /// `q_i = sigma^2 (ln2/h_i)^2` is only comparable across features on a
    /// common scale. Turn it off when the features are already on a sensible
    /// scale and you want the filter to operate on them directly — that makes
    /// this exactly a Bayesian linear regression (with `q = 0` and a fixed
    /// `obs_var`), which is how it is cross-checked against river.
    #[serde(default = "default_true")]
    pub standardize: bool,
    /// Whether `min_weight` holds learning back as well as the output, as
    /// it did before schema 64: a row it withheld had no prediction, so its
    /// residual taught the residual variance `σ²` nothing, and an explicit
    /// `q`, which in units of the noise is read over `σ²`, and a covariance
    /// sized again from the noise followed it. A spec builds `false`:
    /// `min_weight` withholds the output alone, `σ²` learns from the
    /// prediction the filter makes on every row it predicts, and the fit is
    /// the same at any `min_weight` (the module doc). A state written before
    /// schema 64 has no such field and was fitted the old way, so it loads
    /// as `true` and goes on to the bit.
    #[serde(default = "gate_held_learning_before_64")]
    pub min_weight_holds_learning: bool,
    /// Without `obs_var`, hold `P` in units of the noise, so the gain never
    /// reads the noise estimate, and leave the intercept's prior diffuse (the
    /// module doc, "The noise is a scale, not a weight"). Every model the
    /// bank builds sets it; a state saved before it existed has none and
    /// goes on as it was fitted, `P` absolute and the estimate in the gain
    /// (docs/PLAN.md task 278). No effect with `obs_var`, whose `P` is the
    /// same either way.
    #[serde(default)]
    pub noise_units: bool,
    /// The forecasts' horizons, spans of the clock past the row
    /// (`kalman::horizons`); empty, and not written, for none.
    #[serde(default, rename = "forecast_spans")]
    pub horizons: Vec<f64>,
    /// Observation family, link and nonlinear filter settings (`kalman::nonlinear`);
    /// skipped on write when at the linear Gaussian default.
    #[serde(default)]
    pub obs: KalmanObs,
    /// Prior coefficient mean (`n_targets x k_total`, intercept first, original units).
    #[serde(default)]
    pub coef_prior: Option<Vec<Vec<f64>>>,
    /// Emit per-target out-of-sample Gaussian innovation log-likelihood.
    #[serde(default)]
    pub emit_loglik: bool,
    /// Emit per-target out-of-sample innovation Mahalanobis distance.
    #[serde(default)]
    pub emit_mahal: bool,
}

pub use horizons::KALMAN_MAX_HORIZONS;

fn default_true() -> bool {
    true
}

/// What a configuration written before schema 64, which had no
/// [`KalmanCfg::min_weight_holds_learning`], was fitted with: a gate that
/// held learning back too. It must go on as one, to the bit (CLAUDE.md hard
/// rule 5).
fn gate_held_learning_before_64() -> bool {
    true
}

fn default_revert() -> Vec<f64> {
    vec![f64::INFINITY]
}

impl KalmanCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.fit_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        no.take(self.decay.check().map_err(|e| format!("kalman: {e}")));
        no.take(crate::model::check_dims(
            "kalman",
            self.n_features,
            self.n_targets,
        ));
        let k = self.k_total();
        // `the 3 coefficients, intercept included`, as the model has them.
        let coefs = format!(
            "{}{}",
            crate::model::count(k, "coefficient"),
            if self.fit_intercept {
                ", intercept included"
            } else {
                ""
            }
        );
        // Every bad value of a list, not only the first.
        let bad_of = |v: &[f64], ok: &dyn Fn(f64) -> bool| -> Vec<String> {
            v.iter().filter(|x| !ok(**x)).map(f64::to_string).collect()
        };
        // A single value (one for every coefficient) is "it", and the
        // remedy sets it; a list's bad values are named and each is set.
        let one_or_these = |all: &[f64], bad: &[String]| {
            if all.len() == 1 {
                (format!("it is {}", bad[0]), "it")
            } else if let [one] = bad {
                (format!("one of its values is {one}"), "each")
            } else {
                (
                    format!("its values {} are not", crate::model::listed(bad)),
                    "each",
                )
            }
        };
        let first = if self.fit_intercept {
            ", the intercept first"
        } else {
            ""
        };
        let per_coefficient = |key: &str, n: usize| {
            format!(
                "kalman: {key} must list one value, or one for each of the {coefs}, but it \
                 lists {n}. Give one value for all the coefficients, or {k} values, one for \
                 each{first}."
            )
        };
        if let Some(q) = &self.q {
            if q.len() != k {
                no.refuse(format!(
                    "kalman: q must list one value for each of the {coefs}, but it lists {}. \
                     Give one value for each coefficient{first}.",
                    q.len()
                ));
            }
            // NaN passes `v < 0.0` and `v <= 0.0` alike, so each bound in
            // this function names it. A NaN `obs_var` was the silent case:
            // `s_inn` NaN on every row, the update's guard never met, and
            // the filter predicting its prior for the life of the stream
            // with no error and no counted failure (review 2026-09-18, B4).
            // Finite too, as the spec layer has it: an infinite `q` puts
            // `inf` on `P`'s diagonal (review 2026-10-06, CC7).
            let bad = bad_of(q, &|v| v.is_finite() && v >= 0.0);
            if !bad.is_empty() {
                let (said, set) = one_or_these(q, &bad);
                no.refuse(format!(
                    "kalman: q must be finite and at least 0, but {said}. Set {set} to 0 for no \
                     drift over time, or to a finite positive variance."
                ));
            }
        } else if self.obs.is_gaussian_identity() {
            // The spec's `coef_half_life`: the field is the configuration's
            // `half_life`, but a spec's `half_life` is another parameter.
            if self.half_life.len() != 1 && self.half_life.len() != k {
                no.refuse(per_coefficient("coef_half_life", self.half_life.len()));
            }
            let bad = bad_of(&self.half_life, &|h| h > 0.0);
            if !bad.is_empty() {
                let (said, set) = one_or_these(&self.half_life, &bad);
                no.refuse(format!(
                    "kalman: coef_half_life must be above 0, but {said}. Set {set} to a positive \
                     number of clock units, or to inf to hold the coefficient fixed."
                ));
            }
        }
        if self.revert_half_life.len() != 1 && self.revert_half_life.len() != k {
            no.refuse(per_coefficient(
                "revert_half_life",
                self.revert_half_life.len(),
            ));
        }
        let bad = bad_of(&self.revert_half_life, &|h| h > 0.0);
        if !bad.is_empty() {
            let (said, set) = one_or_these(&self.revert_half_life, &bad);
            no.refuse(format!(
                "kalman: revert_half_life must be above 0, but {said}. Set {set} to a positive \
                 number of clock units, or to inf for a coefficient that drifts with no pull \
                 toward 0."
            ));
        }
        // Finite, as the spec layer has them: at `p0 = inf` `P` is never
        // sized and the gain is 0 for good, and at `obs_var = inf` every
        // update is skipped; either predicts 0 for the life of the stream
        // (review 2026-10-06, CC7).
        if !(self.p0.is_finite() && self.p0 > 0.0) {
            no.refuse(format!(
                "kalman: p0 must be finite and above 0, but it is {}. Set p0 to each \
                 slope's starting variance, as a multiple of the target's noise variance, such \
                 as the default 1.",
                self.p0
            ));
        }
        if let Some(v) = self.obs_var.filter(|v| !(v.is_finite() && *v > 0.0)) {
            no.refuse(format!(
                "kalman: obs_var must be finite and above 0, but it is {v}. Set obs_var to the \
                 target's noise variance, or leave it out to estimate it from the residuals."
            ));
        }
        if let Some(prior) = &self.coef_prior {
            no.take(crate::model::check_coef_prior(
                "kalman",
                prior,
                self.n_targets,
                k,
                self.fit_intercept,
            ));
        }
        horizons::check(self, &mut no);
        self.obs.validate(self, &mut no);
        no.take(crate::model::check_min_weight("kalman", self.min_weight));
        no.finish()
    }

    /// Whether any slot reverts (`Phi != I`). The default random walk skips
    /// the transition entirely, so it stays bit-identical to before E41.
    fn reverts(&self) -> bool {
        self.revert_half_life.iter().any(|h| h.is_finite())
    }

    /// Slot `i`'s reversion half-life `r_i`, `inf` for a random walk.
    fn revert_of(&self, i: usize) -> f64 {
        if self.revert_half_life.len() == 1 {
            self.revert_half_life[0]
        } else {
            self.revert_half_life[i]
        }
    }

    /// The transition factor of slot `i` over a clock delta `d`,
    /// `2^(-d/r_i)`, spelled as [`Decay::factor`] is.
    fn phi(&self, i: usize, d_clock: f64) -> f64 {
        Decay::Halflife(self.revert_of(i)).factor(d_clock)
    }

    /// What slot `i`'s process noise `q_i` is multiplied by for an
    /// observation a gap of `d` after the last (the module doc): `d²` on a
    /// random walk, and on a slot reverting at `r_i`, `((1 − 2^(−d/r_i)) /
    /// θ_i)²` with `θ_i = ln 2 / r_i` -- `d²` for a gap well under `r_i`,
    /// and `1/θ_i²` past it, where `d²` grew without bound across a run of
    /// rows that observe nothing (docs/PLAN.md task 214). `1 − 2^(−d/r_i)` is
    /// `−expm1(−θ_i d)`, whole at a small `d`.
    fn gap_noise(&self, i: usize, d: f64) -> f64 {
        let r = self.revert_of(i);
        if r.is_infinite() {
            return d * d;
        }
        let theta = std::f64::consts::LN_2 / r;
        let g = -(-(theta * d)).exp_m1() / theta;
        g * g
    }
}

/// [`KalmanCfg::gap_noise`] of every slot for one gap, kept beside the
/// model with the gap's exact 64 bits: on a regular clock every row that
/// observes its target charges the same gap, and takes the kept factors
/// rather than an `exp_m1` per reverting slot. A value of the bits, so a
/// hit returns what the call would, to the bit; never saved, so a restored
/// model starts empty. One `exp_m1` serves every slot under a single
/// `revert_half_life`, the slots' factor being one number.
#[derive(Debug, Clone, Default)]
struct GapFactors {
    /// The gap's bits, once a row has computed its factors.
    bits: Option<u64>,
    vals: Vec<f64>,
    /// Rows that took the kept factors: what the tests read.
    #[cfg(test)]
    hits: u64,
}

/// A cache, not state: two models are equal whatever either has kept.
impl PartialEq for GapFactors {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl GapFactors {
    /// `cfg.gap_noise(i, d)` for each slot `i`.
    fn of(&mut self, cfg: &KalmanCfg, d: f64) -> &[f64] {
        let k = cfg.k_total();
        let bits = d.to_bits();
        if self.bits == Some(bits) && self.vals.len() == k {
            #[cfg(test)]
            {
                self.hits += 1;
            }
            return &self.vals;
        }
        self.vals.clear();
        if cfg.revert_half_life.len() == 1 {
            // `revert_of` is the one value for every slot.
            let g = cfg.gap_noise(0, d);
            self.vals.resize(k, g);
        } else {
            self.vals.extend((0..k).map(|i| cfg.gap_noise(i, d)));
        }
        self.bits = Some(bits);
        &self.vals
    }
}

/// How far the moments may drift from the anchor before `b` and `P` are
/// re-mapped to them (the module doc): a scale by a factor of `1 + τ`
/// either way, a mean by `τ` of the anchor's scale.
const ANCHOR_DRIFT: f64 = 1.0;

/// How many rows' squared innovations a standardized prior rests on before
/// it is set (the module doc).
const PRIOR_ROWS: usize = 3;

/// The median of a χ²₁, `0.4549…`: what the median of a Gaussian
/// innovation's square is, in units of its variance, so the median of the
/// first rows' squares over it reads the noise variance, as a mean of them
/// does (the module doc; docs/PLAN.md task 214).
const CHI2_1_MEDIAN: f64 = 0.454_936_423_119_572_8;

/// The squared innovations a covariance's prior is sized from (the module
/// doc): the first [`PRIOR_ROWS`] rows that observed one of its targets,
/// each row's mean squared innovation over the targets it observed and its
/// weight. Kept under `standardize` without `obs_var`, and empty otherwise.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct NoiseBasis {
    e2: Vec<f64>,
    w: Vec<f64>,
}

impl NoiseBasis {
    /// A row's `n` squared innovations, summing to `e2`, at weight `w`: one
    /// row, whatever the number of targets it observes, so a target beside
    /// an exact copy of itself under `share_p` sizes its prior on the row it
    /// would alone. The rows past the first [`PRIOR_ROWS`] are not kept.
    fn add(&mut self, e2: f64, n: f64, w: f64) {
        let v = e2 / n;
        if self.e2.len() < PRIOR_ROWS && v > 0.0 && v.is_finite() && w > 0.0 && w.is_finite() {
            self.e2.push(v);
            self.w.push(w);
        }
    }

    /// The weighted median of the first rows' squared innovations over
    /// [`CHI2_1_MEDIAN`], once there are [`PRIOR_ROWS`] of them; none
    /// before. A median and not a mean (docs/PLAN.md task 214): a target at
    /// the input bound among the first rows, a square of `1e200`, made the
    /// mean `3e199`, and the prior sized from it left `P` indefinite for
    /// good.
    fn noise(&self) -> Option<f64> {
        if self.e2.len() < PRIOR_ROWS {
            return None;
        }
        // On the stack: read on every row, a vector a row was a tenth of
        // `kalman`'s bank overhead (task 250). The same stable sort over the
        // same pairs, so the same median.
        let mut held = [(0.0, 0.0); PRIOR_ROWS];
        let rows = &mut held[..self.e2.len().min(self.w.len()).min(PRIOR_ROWS)];
        for (r, (&e, &w)) in rows.iter_mut().zip(self.e2.iter().zip(&self.w)) {
            *r = (e, w);
        }
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        let half = 0.5 * self.w.iter().sum::<f64>();
        let mut seen = 0.0;
        let median = rows
            .iter()
            .find(|(_, w)| {
                seen += w;
                seen >= half
            })
            .map_or(rows[rows.len() - 1].0, |(v, _)| *v);
        let v = median / CHI2_1_MEDIAN;
        (v > 0.0 && v.is_finite()).then_some(v)
    }

    fn is_valid(&self) -> bool {
        self.e2.len() == self.w.len()
            && self.e2.len() <= PRIOR_ROWS
            && self
                .e2
                .iter()
                .chain(&self.w)
                .all(|v| v.is_finite() && *v > 0.0)
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(try_from = "KalmanV4")]
pub struct Kalman {
    cfg: KalmanCfg,
    /// Standardization stats over `z` (shared across targets): the means and
    /// variances the scales are read from.
    stats: EwDiag,
    /// The anchor (the module doc): per slot, the mean as a pair and the
    /// scale of the moments at the last re-map, the coordinates `b` and `P`
    /// are held in. Empty without `standardize`; the intercept's slot is 0,
    /// 0 and 1.
    anchor_hi: Vec<f64>,
    anchor_lo: Vec<f64>,
    anchor_scale: Vec<f64>,
    /// Per target: coefficient mean in the anchor's coordinates.
    beta: Vec<Vec<f64>>,
    /// Per target (or one when `share_p`): covariance, row-major `k*k`, in
    /// the anchor's coordinates; a slot with a variance of 0 is unsized
    /// (the module doc).
    p: Vec<Vec<f64>>,
    /// Per covariance: the clock since it last took a row that observes its
    /// target (the module doc, task 211).
    elapsed: Vec<f64>,
    /// The moments' own gap: the clock since the last row that moved them,
    /// a row of weight above 0, whether it observed a target or not (the
    /// module doc, *After a long gap*). Written only where it is not the
    /// clock since a target was last observed (review 2026-11-23, KAL-6).
    moments_gap: f64,
    /// Per covariance: the squared innovations its prior is sized from.
    basis: Vec<NoiseBasis>,
    /// EW residual variance per target and its weight sum.
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    wj: Vec<f64>,
    #[serde(skip)]
    zbuf: Vec<f64>,
    #[serde(skip)]
    zs: Vec<f64>,
    #[serde(skip)]
    pz: Vec<f64>,
    #[serde(skip)]
    gain: Vec<f64>,
    /// This row's `z . b_j` per target, before any update: the prediction,
    /// and the innovation the update and the first noise read.
    #[serde(skip)]
    zb: Vec<f64>,
    /// This row's transition factors, one per slot (only filled when a
    /// slot reverts), and the transition's coupling `g` in the anchor's
    /// coordinates.
    #[serde(skip)]
    phi: Vec<f64>,
    #[serde(skip)]
    gbuf: Vec<f64>,
    /// This row's process-noise variances, kept between rows so a step
    /// allocates nothing for them (docs/records/PERFORMANCE-LOG.md §13).
    #[serde(skip)]
    qbuf: Vec<f64>,
    /// `share_p`'s noises, sorted to be summed in ascending order.
    #[serde(skip)]
    sumbuf: Vec<f64>,
    /// This row's noise per target, `R_j` and the `sigma^2` of `Q`.
    #[serde(skip)]
    noise: Vec<f64>,
    /// Which slots hold anything, in any target or covariance.
    #[serde(skip)]
    live: Vec<bool>,
    /// The map from the anchor to the moments as they stand, `a_i` and `c_i`
    /// (the module doc), and whether it was formed from the moments as they
    /// stand: formed after each row that moves them, for the next.
    #[serde(skip)]
    map_a: Vec<f64>,
    #[serde(skip)]
    map_c: Vec<f64>,
    #[serde(skip)]
    map_fresh: bool,
    /// The re-map's row `u = (Mᵀ P)_0·` for each `P`, and the transition's
    /// `P g`.
    #[serde(skip)]
    ubuf: Vec<Vec<f64>>,
    #[serde(skip)]
    vbuf: Vec<f64>,
    /// The gap noise's factors for the last gap charged ([`GapFactors`]).
    #[serde(skip)]
    gap_factors: GapFactors,
    /// The form `p` is held and updated in (`kalman/forms.rs`): `P`, or
    /// its square root's factor. Written only where it is not the default,
    /// so no state of the covariance form moves (schema 68), or where the
    /// moments' gap is written after it.
    form: CovarianceForm,
    /// The factor's `f = Cᵀ z` and a slot's `x`.
    #[serde(skip)]
    fbuf: Vec<f64>,
    #[serde(skip)]
    xbuf: Vec<f64>,
    /// The fixed-lag smoother (`kalman/smooth.rs`), written only where one
    /// was asked for, so no other state moves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    smoother: Option<Box<smooth::Smoother>>,
    #[serde(skip)]
    nl_work: Option<Box<nonlinear::NonlinearWork>>,
}

/// The layout `Kalman` loads, checked on the way in: every vector at the
/// cfg's width, so a damaged state is refused rather than panicking on its
/// first step. (A schema-2 file kept a full `EwCov` under `cov` and was
/// converted by taking its diagonal; the minimum schema has passed it,
/// docs/PLAN.md tasks 198 and 194-202.)
#[derive(Deserialize)]
struct KalmanV4 {
    cfg: KalmanCfg,
    stats: EwDiag,
    anchor_hi: Vec<f64>,
    anchor_lo: Vec<f64>,
    anchor_scale: Vec<f64>,
    beta: Vec<Vec<f64>>,
    p: Vec<Vec<f64>>,
    elapsed: Vec<f64>,
    basis: Vec<NoiseBasis>,
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    wj: Vec<f64>,
    /// A state of schema 67 or earlier has none: the covariance form.
    #[serde(rename = "covariance_form", default)]
    form: CovarianceForm,
    /// A state without it -- one of schema 71 or earlier, or one whose last
    /// row that moved the moments observed a target -- reads the clock
    /// since a target was last observed (review 2026-11-23, KAL-6).
    #[serde(default)]
    moments_gap: Option<f64>,
    /// A state without one has no smoother.
    #[serde(default)]
    smoother: Option<Box<smooth::Smoother>>,
}

/// How [`Kalman`] is written: its fields in [`KalmanV4`]'s order, the
/// covariance form only where it is not the default, and the moments' gap
/// next, only where it is not the clock since a target was last observed,
/// and the smoother last, only where one was asked for. Where a later field
/// is written so is every one before it (the gap as `None` where it is the
/// default), so the positional encoding, which has no names, never reads one
/// in the other's place.
impl Serialize for Kalman {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let gap = self.written_gap();
        let smoother = self.smoother.as_ref();
        let form = gap.is_some() || smoother.is_some() || !self.form.is_covariance();
        let write_gap = gap.is_some() || smoother.is_some();
        let n = 12 + usize::from(form) + usize::from(write_gap) + usize::from(smoother.is_some());
        let mut st = s.serialize_struct("Kalman", n)?;
        st.serialize_field("cfg", &self.cfg)?;
        st.serialize_field("stats", &self.stats)?;
        st.serialize_field("anchor_hi", &self.anchor_hi)?;
        st.serialize_field("anchor_lo", &self.anchor_lo)?;
        st.serialize_field("anchor_scale", &self.anchor_scale)?;
        st.serialize_field("beta", &self.beta)?;
        st.serialize_field("p", &self.p)?;
        st.serialize_field("elapsed", &self.elapsed)?;
        st.serialize_field("basis", &self.basis)?;
        st.serialize_field("sig2", &self.sig2)?;
        st.serialize_field("wsig", &self.wsig)?;
        st.serialize_field("wj", &self.wj)?;
        if form {
            st.serialize_field("covariance_form", &self.form)?;
        } else {
            st.skip_field("covariance_form")?;
        }
        if write_gap {
            st.serialize_field("moments_gap", &gap)?;
        } else {
            st.skip_field("moments_gap")?;
        }
        match smoother {
            Some(sm) => st.serialize_field("smoother", sm)?,
            None => st.skip_field("smoother")?,
        }
        st.end()
    }
}

impl TryFrom<KalmanV4> for Kalman {
    type Error = String;

    fn try_from(v: KalmanV4) -> Result<Self, String> {
        let cfg = v.cfg;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        let n_p = if cfg.share_p { 1 } else { m };
        let anchored = if cfg.standardize { k } else { 0 };
        if v.stats.k() != k
            || v.beta.len() != m
            || v.beta.iter().any(|b| b.len() != k)
            || v.p.len() != n_p
            || v.p.iter().any(|p| p.len() != k * k)
            || v.elapsed.len() != n_p
            || v.basis.len() != n_p
            || v.sig2.len() != m
            || v.wsig.len() != m
            || v.wj.len() != m
        {
            return Err(
                "kalman: its saved state has the wrong shape. Load an undamaged copy of \
                        the state file, or rebuild it from its input."
                    .into(),
            );
        }
        // The anchor is the standardizer's: without one there is no
        // coordinate to hold (docs/PLAN.md task 211).
        if v.anchor_hi.len() != anchored
            || v.anchor_lo.len() != anchored
            || v.anchor_scale.len() != anchored
        {
            let set = if cfg.standardize { "True" } else { "False" };
            return Err(format!(
                "kalman: its saved feature means and scales do not match its setting \
                 standardize={set}. Load an undamaged copy of the state file, or rebuild it from its \
                 input."
            ));
        }
        if !v.anchor_scale.iter().all(|s| s.is_finite() && *s > 0.0)
            || !v
                .anchor_hi
                .iter()
                .chain(&v.anchor_lo)
                .all(|m| m.is_finite())
        {
            return Err(
                "kalman: its saved feature means and scales hold values no fit could \
                 have. Load an undamaged copy of the state file, or rebuild it from its input."
                    .into(),
            );
        }
        if !v.elapsed.iter().all(|e| e.is_finite() && *e >= 0.0)
            || v.moments_gap.is_some_and(|g| !(g.is_finite() && g >= 0.0))
        {
            return Err(
                "kalman: its saved clocks since each observation hold values no fit \
                 could have. Load an undamaged copy of the state file, or rebuild it from its \
                 input."
                    .into(),
            );
        }
        let moments_gap = v.moments_gap.unwrap_or_else(|| unobserved(&v.elapsed));
        // Each message says what its check requires (docs/PLAN.md task 218):
        // the basis's said ">= 0" of a check that refuses a 0, and named
        // neither its lengths nor its bound.
        if v.form != CovarianceForm::Covariance && !cfg.obs.is_linear() {
            return Err(
                "kalman: its saved covariance_form is not supported with a nonlinear filter. Load \
                 an undamaged copy of the state file, or rebuild it from its input."
                    .into(),
            );
        }
        if v.smoother.as_ref().is_some_and(|s| {
            v.form != CovarianceForm::Covariance || !s.is_valid(k, m, n_p, cfg.standardize)
        }) {
            return Err(
                "kalman: its saved smoother has the wrong shape. Load an undamaged copy of the \
                 state file, or rebuild it from its input."
                    .into(),
            );
        }
        if !v.basis.iter().all(NoiseBasis::is_valid) {
            return Err(
                "kalman: its saved prior rows hold values no fit could have. Load an \
                 undamaged copy of the state file, or rebuild it from its input."
                    .into(),
            );
        }
        // The scratch buffers are sized by `ensure_buffers` on the first use.
        Ok(Self {
            cfg,
            stats: v.stats,
            anchor_hi: v.anchor_hi,
            anchor_lo: v.anchor_lo,
            anchor_scale: v.anchor_scale,
            beta: v.beta,
            p: v.p,
            elapsed: v.elapsed,
            moments_gap,
            basis: v.basis,
            sig2: v.sig2,
            wsig: v.wsig,
            wj: v.wj,
            zbuf: vec![],
            zs: vec![],
            pz: vec![],
            gain: vec![],
            zb: vec![],
            phi: vec![],
            gbuf: vec![],
            qbuf: vec![],
            sumbuf: vec![],
            noise: vec![],
            live: vec![],
            map_a: vec![],
            map_c: vec![],
            map_fresh: false,
            ubuf: vec![],
            vbuf: vec![],
            gap_factors: GapFactors::default(),
            form: v.form,
            fbuf: vec![],
            xbuf: vec![],
            smoother: v.smoother,
            nl_work: None,
        })
    }
}

impl Kalman {
    pub fn new(cfg: KalmanCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        let n_p = if cfg.share_p { 1 } else { m };
        let off = usize::from(cfg.fit_intercept);
        // Unsized, all zero, until a row sizes the noise; with `obs_var` the
        // noise is known now, and so is the prior: of every slot
        // unstandardized, and standardized of the intercept, each feature's
        // waiting for its scale (the module doc).
        let mut p_init = vec![0.0; k * k];
        if !cfg.obs.is_gaussian_identity() {
            let sized = if cfg.standardize { off } else { k };
            for i in 0..sized {
                p_init[i * k + i] = cfg.p0;
            }
        } else if let Some(r) = cfg.obs_var {
            let sized = if cfg.standardize { off } else { k };
            for i in 0..sized {
                p_init[i * k + i] = cfg.p0 * r;
            }
        } else if cfg.noise_units && !cfg.standardize {
            // In units of the noise the prior needs no noise: `p0` on each
            // slope now, the intercept left diffuse for its first row.
            for i in off..k {
                p_init[i * k + i] = cfg.p0;
            }
        }
        let beta = innovation::initial_beta(&cfg);
        let anchored = if cfg.standardize { k } else { 0 };
        let mut model = Self {
            stats: EwDiag::new(k),
            anchor_hi: vec![0.0; anchored],
            anchor_lo: vec![0.0; anchored],
            anchor_scale: vec![1.0; anchored],
            beta,
            p: vec![p_init; n_p],
            elapsed: vec![0.0; n_p],
            moments_gap: 0.0,
            basis: vec![NoiseBasis::default(); n_p],
            sig2: vec![0.0; m],
            wsig: vec![0.0; m],
            wj: vec![0.0; m],
            zbuf: vec![],
            zs: vec![],
            pz: vec![],
            gain: vec![],
            zb: vec![],
            phi: vec![],
            gbuf: vec![],
            qbuf: vec![],
            sumbuf: Vec::with_capacity(m),
            noise: vec![],
            live: vec![],
            map_a: vec![],
            map_c: vec![],
            map_fresh: false,
            ubuf: vec![],
            vbuf: vec![],
            gap_factors: GapFactors::default(),
            form: CovarianceForm::Covariance,
            fbuf: vec![],
            xbuf: vec![],
            smoother: None,
            nl_work: None,
            cfg,
        };
        model.ensure_buffers();
        Ok(model)
    }

    pub fn cfg(&self) -> &KalmanCfg {
        &self.cfg
    }

    pub fn sigma2(&self) -> &[f64] {
        &self.sig2
    }

    /// Variance of the *last* prediction, per target: `zᵀ P_j z + R_j` —
    /// parameter uncertainty plus observation noise (ENHANCEMENTS E12).
    ///
    /// `P` here is the covariance as it stands, so this is the **filtered**
    /// variance at the last regressor `z`, not the one-step-ahead predictive
    /// variance: that would carry `P` through the transition and add the
    /// process noise for the clock since the last observation, `zᵀ(Φ P Φᵀ +
    /// Q·D²)z + R`. The two differ by `zᵀ(Φ P Φᵀ − P + Q·D²)z`, which the
    /// default random walk (`Φ = I`) reduces to `Q·D²`: negligible under a
    /// half-life-derived `q` (`q/R = (ln2/h)²`, 0.005 % at half-life 100 and
    /// unit spacing, `D²` times that at spacing `D`), not under a large
    /// explicit `q` (review 2026-09-18, D2).
    ///
    /// This is the piece `sigma` alone cannot give. `sigma` is the spread of
    /// realized errors; this also knows how unsure the filter is about its own
    /// coefficients, so it is wide during warmup and after a gap, and narrows
    /// as evidence accumulates. Only Kalman tracks `P`, so only Kalman can
    /// report it exactly, under its model (the module doc).
    ///
    /// For the row `x`, standardized as `predict` standardizes it. It read
    /// the regressor of the last row stepped, a scratch a save does not
    /// keep, so a loaded filter answered for no regressor at all -- `R` alone
    /// (review 2026-09-12, V7).
    ///
    /// NaN until the target has a residual variance. The noise `step` uses
    /// before then is the row's own innovation squared, and a prediction is
    /// made before its row's target is seen, so it has no innovation to read
    /// (review 2026-10-05, CC4). Under `share_p` target `j`'s covariance is
    /// the shared `P` times `σ²_j / σ̄²` (the module doc, task 211), so this
    /// is `σ²_j (zᵀ P z / σ̄² + 1)`. After a row that observes nothing, `P`
    /// as it stands carries the noise of the clock since the last
    /// observation, `Q·D²`, as the next observation will charge it (the
    /// module doc).
    pub fn pred_var(&self, x: &[f64]) -> Vec<f64> {
        if !self.cfg.obs.is_gaussian_identity() {
            return self.pred_var_nonlinear(x);
        }
        let k = self.cfg.k_total();
        let z = self.anchored(x);
        (0..self.cfg.n_targets)
            .map(|j| {
                let pi = if self.cfg.share_p { 0 } else { j };
                let p = self.covariance_now(pi);
                let p = &p[..];
                let mut quad = 0.0;
                for i in 0..k {
                    let row = i * k;
                    let mut acc = 0.0;
                    for jj in 0..k {
                        acc += p[row + jj] * z[jj];
                    }
                    quad += z[i] * acc;
                }
                self.variance_of(j, quad)
            })
            .collect()
    }

    /// Whether the moments' update on a row `gap` clock units after the
    /// moments last moved keeps the row's own weight of history (`step_as`,
    /// `EwDiag::update_keeping`): under `standardize`, more than one
    /// half-life on, read from the clock alone.
    pub(crate) fn keeps_history(&self, gap: f64) -> bool {
        self.cfg.standardize && self.cfg.decay.past_half_life(gap)
    }

    /// The moments' gap as the last row left it: the clock since the last
    /// row of weight above 0 (the module doc, *After a long gap*). The next
    /// row's floor reads it with that row's step added.
    #[cfg(test)]
    pub(crate) fn moments_gap(&self) -> f64 {
        self.moments_gap
    }

    /// The moments' gap where a state writes it: where it is not the clock
    /// since a target was last observed, which a state without it reads.
    fn written_gap(&self) -> Option<f64> {
        (self.moments_gap.to_bits() != unobserved(&self.elapsed).to_bits())
            .then_some(self.moments_gap)
    }

    pub fn n_eff(&self) -> f64 {
        self.stats.n_eff()
    }

    /// Feature scales of the moments as they stand: sd for features, 1 for
    /// the intercept slot. Zero-variance features get scale 1 (their
    /// standardized value is then their centered value, i.e. 0). All ones
    /// when `standardize` is off.
    fn scales(&self) -> Vec<f64> {
        (0..self.cfg.k_total())
            .map(|i| self.current_scale(i).0)
            .collect()
    }

    /// Slot `i`'s scale under the moments as they stand, and whether it is
    /// usable: a variance above 0 with an intercept, a raw second moment
    /// above 0 through the origin (review 2026-09-12, C10), and 1, usable,
    /// for the intercept and without `standardize`.
    fn current_scale(&self, i: usize) -> (f64, bool) {
        let off = usize::from(self.cfg.fit_intercept);
        if !self.cfg.standardize || i < off {
            (1.0, true)
        } else if off == 0 {
            // No intercept: nothing to centre on, so the scale is the raw
            // second moment's -- any positive one is usable, there being
            // no cancellation.
            let raw = self.stats.raw(i);
            if raw > 0.0 && raw.is_finite() {
                (raw.sqrt(), true)
            } else {
                (1.0, false)
            }
        } else {
            let v = self.stats.var(i);
            let raw = self.stats.raw(i);
            if crate::variance_is_usable(v, raw) {
                (v.sqrt(), true)
            } else {
                (1.0, false)
            }
        }
    }

    /// Coefficients in the ORIGINAL feature units, per target.
    ///
    /// Under `standardize` the filter's state lives in the anchor's
    /// coordinates, and it is read out through the anchor's means and
    /// scales: `c_i = b_i / s^a_i`, `c_0 = b_0 − Σ_i c_i m^a_i`. These are
    /// the coefficients `pred` uses, and they move only with the fit, to
    /// rounding: a move of the moments is a change of coordinates, which the
    /// state follows (the module doc; docs/PLAN.md tasks 206 and 211).
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        if !self.cfg.standardize {
            return self.beta.clone();
        }
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let mut out = Vec::with_capacity(self.cfg.n_targets);
        for b in &self.beta {
            let mut c = vec![0.0; k];
            for (i, ci) in c.iter_mut().enumerate().skip(off) {
                *ci = b[i] / self.anchor_scale[i];
            }
            if self.cfg.fit_intercept {
                // b0 is on centred features: unshift by the anchor's means.
                let mut b0 = b[0];
                for (i, ci) in c.iter().enumerate().skip(off) {
                    b0 -= ci * self.anchor_hi[i];
                    b0 -= ci * self.anchor_lo[i];
                }
                c[0] = b0;
            }
            out.push(c);
        }
        out
    }

    /// Process-noise variances for this row, `q_i = sigma^2 * (ln2 / h_i)^2`
    /// (steady-state gain matching with EW-RLS on standardized features).
    #[cfg(test)]
    fn q_vec(&self, sigma2: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.cfg.k_total()];
        self.q_into(sigma2, &mut out);
        out
    }

    /// [`Self::q_vec`] into a caller's buffer of length `k_total`.
    fn q_into(&self, sigma2: f64, out: &mut [f64]) {
        if let Some(q) = &self.cfg.q {
            out.copy_from_slice(q);
            return;
        }
        let shared = self.cfg.half_life.len() == 1;
        for (i, qi) in out.iter_mut().enumerate() {
            let h = self.cfg.half_life[if shared { 0 } else { i }];
            *qi = if h.is_infinite() {
                0.0
            } else {
                let r = std::f64::consts::LN_2 / h;
                sigma2 * r * r
            };
        }
    }

    /// Whether `P` is held in units of the noise (the module doc): the
    /// noise learned, under [`KalmanCfg::noise_units`].
    fn units(&self) -> bool {
        self.cfg.obs.is_gaussian_identity() && self.cfg.noise_units && self.cfg.obs_var.is_none()
    }

    /// The process noise `P` takes for a noise `sigma2`, in the units `P` is
    /// held in, into `out`: [`Self::q_into`]'s variances, or in units of the
    /// noise those over `sigma2` -- a derived `q` is `sigma2` times a
    /// number, so that number; an explicit `q` over `sigma2`, and none where
    /// there is no noise to measure it against.
    fn q_held_into(&self, sigma2: f64, out: &mut [f64]) {
        if !self.units() {
            self.q_into(sigma2, out);
        } else if self.cfg.q.is_none() {
            self.q_into(1.0, out);
        } else if sigma2 > 0.0 && sigma2.is_finite() {
            self.q_into(sigma2, out);
            for qi in out.iter_mut() {
                *qi /= sigma2;
            }
        } else {
            out.fill(0.0);
        }
    }

    /// Whether covariance `pi`'s intercept is diffuse, waiting for its first
    /// row (the module doc): in units of the noise, its row and column 0
    /// until that row's exact step. Two writes would reach them meanwhile,
    /// and are kept off: the process noise unstandardized, where every slot
    /// takes it, on a first observed row after a gap; and a slope's prior
    /// carried into an anchor another covariance's slopes have moved, its
    /// cross terms with the intercept. Otherwise the step would find a small
    /// variance there, and the intercept would start at 0 as a prior sure of
    /// it.
    fn open<const FACTOR: bool>(&self, pi: usize) -> bool {
        self.units() && self.cfg.fit_intercept && !self.sized::<FACTOR>(pi, 0)
    }

    /// A quadratic form in `P`, `z' P z` or a trace, over the noise `r`: in
    /// units of the noise it already is.
    fn per_noise(&self, quad: f64, r: f64) -> f64 {
        if self.units() { quad } else { quad / r }
    }

    /// `[1, x]` standardized against the moments as they stand: the current
    /// coordinates (the module doc).
    fn standardized(&self, x: &[f64]) -> Vec<f64> {
        let off = usize::from(self.cfg.fit_intercept);
        (0..self.cfg.k_total())
            .map(|i| {
                let raw = if i < off { 1.0 } else { x[i - off] };
                if !self.cfg.standardize || i < off {
                    raw
                } else if off == 0 {
                    // No intercept to absorb a shift: scale only (C10).
                    raw / self.current_scale(i).0
                } else {
                    self.stats.deviation(i, raw) / self.current_scale(i).0
                }
            })
            .collect()
    }

    /// `[1, x]` in the anchor's coordinates, as `step` reads it: a slot that
    /// holds anything against the anchor, one that holds nothing against the
    /// moments as they stand (`step` sets such a slot's anchor to them before
    /// it holds anything; the module doc).
    fn anchored(&self, x: &[f64]) -> Vec<f64> {
        let off = usize::from(self.cfg.fit_intercept);
        let mut z = vec![1.0; self.cfg.k_total()];
        z[off..].copy_from_slice(x);
        let live: Vec<bool> = (0..z.len()).map(|i| self.is_live(i)).collect();
        let mut out = vec![0.0; z.len()];
        self.anchored_into(&z, &live, &mut out);
        out
    }

    /// The row `z` (the intercept's 1 included) in the anchor's coordinates
    /// into `out`, `live` saying which slots hold anything.
    fn anchored_into(&self, z: &[f64], live: &[bool], out: &mut [f64]) {
        let off = usize::from(self.cfg.fit_intercept);
        for (i, o) in out.iter_mut().enumerate() {
            *o = if !self.cfg.standardize || i < off {
                z[i]
            } else if live[i] {
                if off == 0 {
                    z[i] / self.anchor_scale[i]
                } else {
                    crate::comp::dev(z[i], self.anchor_hi[i], self.anchor_lo[i])
                        / self.anchor_scale[i]
                }
            } else if off == 0 {
                z[i] / self.current_scale(i).0
            } else {
                self.stats.deviation(i, z[i]) / self.current_scale(i).0
            };
        }
    }

    /// Whether slot `i` holds anything: a coefficient or a variance, in any
    /// target or covariance. A slot that holds nothing reads no coordinate.
    fn is_live(&self, i: usize) -> bool {
        if self.factor() {
            self.is_live_as::<true>(i)
        } else {
            self.is_live_as::<false>(i)
        }
    }

    /// [`Kalman::is_live`] in the form `FACTOR` names (`Kalman::sized`).
    #[inline(always)]
    fn is_live_as<const FACTOR: bool>(&self, i: usize) -> bool {
        let sized = if FACTOR {
            (0..self.p.len()).any(|pi| self.sized::<true>(pi, i))
        } else {
            let k = self.cfg.k_total();
            self.p.iter().any(|p| p[i * k + i] != 0.0)
        };
        sized || self.beta.iter().any(|b| b[i] != 0.0)
    }

    /// Every slot's [`Kalman::is_live`], on every row a standardized filter
    /// takes, in the form `FACTOR` names: read once a row, not once a slot
    /// and covariance, which cost the covariance form 5% of its row (task
    /// 281 (3)'s budget).
    fn mark_live<const FACTOR: bool>(&mut self) {
        let mut live = std::mem::take(&mut self.live);
        for (i, l) in live.iter_mut().enumerate() {
            *l = self.is_live_as::<FACTOR>(i);
        }
        self.live = live;
    }

    /// The map from the anchor to the moments as they stand, `a_i = s_i /
    /// s^a_i` and `c_i = (m_i − m^a_i) / s^a_i` (0 without an intercept),
    /// into `a` and `c`, and the drift over the slots `live` marks, `max(a_i
    /// − 1, 1/a_i − 1, |c_i|)` (the module doc). The intercept's slot is 1
    /// and 0.
    ///
    /// Written across the slots: each slot's scale, `a_i` and `c_i` in one
    /// loop with no branch the compiler cannot turn into a select, so it
    /// runs as a vector loop, then the drift over the live slots in slot
    /// order. Every row forms it, and a slot at a time it was about 70 ns of
    /// a 600 ns row at `k = 20`, a square root and two divisions a slot;
    /// across them, about 20 ns less (task 250). The same operations on the
    /// same numbers as [`Self::current_scale`] and the divisions by the
    /// anchor's scale, so the same bits: a usable variance there is
    /// `clamp_rounding(v) > 0` and finite, which is `v > 0` and finite.
    fn map_into(&self, live: &[bool], a: &mut [f64], c: &mut [f64]) -> f64 {
        let k = a.len();
        if !self.cfg.standardize {
            a.fill(1.0);
            c.fill(0.0);
            return 0.0;
        }
        let off = usize::from(self.cfg.fit_intercept);
        a[..off].fill(1.0);
        c[..off].fill(0.0);
        let (m, m_lo, var) = self.stats.parts();
        let (m, m_lo, var) = (&m[off..k], &m_lo[off..k], &var[off..k]);
        let sa = &self.anchor_scale[off..k];
        let (a_s, c_s) = (&mut a[off..k], &mut c[off..k]);
        if off == 1 {
            let (hi, lo) = (&self.anchor_hi[1..k], &self.anchor_lo[1..k]);
            for j in 0..k - 1 {
                let vj = var[j];
                let s = if vj > 0.0 && vj.is_finite() {
                    vj.sqrt()
                } else {
                    1.0
                };
                a_s[j] = s / sa[j];
                c_s[j] = ((m[j] - hi[j]) + (m_lo[j] - lo[j])) / sa[j];
            }
        } else {
            for j in 0..k {
                let raw = var[j] + m[j] * m[j];
                let s = if raw > 0.0 && raw.is_finite() {
                    raw.sqrt()
                } else {
                    1.0
                };
                a_s[j] = s / sa[j];
                c_s[j] = 0.0;
            }
        }
        let mut drift = 0.0f64;
        for i in off..k {
            if live[i] {
                drift = drift.max(a[i] - 1.0).max(1.0 / a[i] - 1.0).max(c[i].abs());
            }
        }
        drift
    }

    /// [`Self::map_into`] into the kept map, marked as formed from the
    /// moments as they stand; returns the drift.
    fn form_map(&mut self) -> f64 {
        let (mut a, mut c) = (
            std::mem::take(&mut self.map_a),
            std::mem::take(&mut self.map_c),
        );
        let drift = self.map_into(&self.live, &mut a, &mut c);
        (self.map_a, self.map_c) = (a, c);
        self.map_fresh = true;
        drift
    }

    /// Slot `i`'s anchor set to the moments as they stand, and its map to
    /// 1 and 0.
    fn anchor_slot_here(&mut self, i: usize) {
        let off = usize::from(self.cfg.fit_intercept);
        if i < off {
            return;
        }
        (self.anchor_hi[i], self.anchor_lo[i]) = if off == 1 {
            self.stats.mean_pair(i)
        } else {
            (0.0, 0.0)
        };
        self.anchor_scale[i] = self.current_scale(i).0;
        self.map_a[i] = 1.0;
        self.map_c[i] = 0.0;
    }

    fn ensure_buffers(&mut self) {
        let k = self.cfg.k_total();
        if self.zbuf.len() != k {
            self.zbuf = vec![0.0; k];
            self.zs = vec![0.0; k];
            self.pz = vec![0.0; k];
            self.gain = vec![0.0; k];
            self.zb = vec![0.0; self.cfg.n_targets];
            self.phi = vec![1.0; k];
            self.gbuf = vec![0.0; k];
            self.qbuf = vec![0.0; k];
            self.noise = vec![0.0; self.cfg.n_targets];
            self.live = vec![false; k];
            self.map_a = vec![1.0; k];
            self.map_c = vec![0.0; k];
            self.map_fresh = false;
            self.ubuf = vec![vec![0.0; k]; self.p.len()];
            self.vbuf = vec![0.0; k];
            self.fbuf = vec![0.0; k];
            self.xbuf = vec![0.0; k];
        }
    }

    /// After a row that moved the moments, the drift from the anchor; past
    /// [`ANCHOR_DRIFT`], `b` and `P` re-mapped to the moments as they stand
    /// and the anchor set to them (the module doc). A re-map is refused whole
    /// where a factor `a_i` of a slot that holds anything is outside
    /// `[1/1024, 1024]` or a shift `|c_i|` above 1024 ([`REMAP_LIMIT`]), or
    /// where a number it would leave is not finite (each `b'`, `P'`'s
    /// diagonal and `P'_00`, which bound the rest of a positive semi-definite
    /// `P'`): `b` and `P` then keep their numbers, read in the new
    /// coordinates (docs/IMPROVEMENTS.md C2). The anchor moves either way. A
    /// slot that holds nothing is mapped by 1 and 0: its numbers are 0
    /// whatever its anchor.
    fn follow_the_moments(&mut self) {
        let drift = self.form_map();
        if drift <= ANCHOR_DRIFT {
            return;
        }
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let mut within = true;
        for i in off..k {
            if self.live[i] {
                let (a, c) = (self.map_a[i], self.map_c[i]);
                within &= (1.0 / REMAP_LIMIT..=REMAP_LIMIT).contains(&a) && c.abs() <= REMAP_LIMIT;
            } else {
                self.map_a[i] = 1.0;
                self.map_c[i] = 0.0;
            }
        }
        // The smoother's chain does not cross a re-map refused.
        if !(within && self.remap())
            && let Some(s) = self.smoother.as_deref_mut()
        {
            s.cut(None);
        }
        for i in off..k {
            self.anchor_slot_here(i);
        }
    }

    /// `b <- Mᵀ b`, `P <- Mᵀ P M` from the kept map, unless a number it
    /// would leave is not finite (the module doc). In place, every check
    /// made before anything moves; whether it moved them.
    fn remap(&mut self) -> bool {
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let (a, c) = (&self.map_a, &self.map_c);
        // First the numbers the check reads, formed as the re-map forms them:
        // each `b'`, each `P'`'s diagonal and, with an intercept, `u = (Mᵀ
        // P)_0·` and `P'_00` (kept in `u[0]`). `Mᵀ P M` is positive
        // semi-definite as `P` is, so its other entries are bounded by its
        // diagonal's.
        let mut finite = true;
        for b in &self.beta {
            let mut b0 = b[0];
            for i in off..k {
                finite &= (b[i] * a[i]).is_finite();
                b0 += c[i] * b[i];
            }
            finite &= off == 0 || b0.is_finite();
        }
        let factor = self.form == CovarianceForm::SquareRoot;
        for (p, u) in self.p.iter_mut().zip(self.ubuf.iter_mut()) {
            if factor {
                // `C <- A C`: row 0 `C_0· + Σ c_l C_l·`, row `i` `a_i C_i·`.
                finite &= forms::factor_remap(p, u, a, c, off, false);
                continue;
            }
            for i in off..k {
                finite &= (p[i * k + i] * (a[i] * a[i])).is_finite();
            }
            if off == 1 {
                // u = row 0 of Mᵀ P, the rows of P contiguous.
                u.copy_from_slice(&p[0..k]);
                for l in 1..k {
                    let cl = c[l];
                    for (uj, plj) in u.iter_mut().zip(&p[l * k..(l + 1) * k]) {
                        *uj += cl * plj;
                    }
                }
                let mut p00 = u[0];
                for j in 1..k {
                    p00 += c[j] * u[j];
                }
                u[0] = p00;
                finite &= p00.is_finite();
            }
        }
        if !finite {
            return false;
        }
        for b in &mut self.beta {
            if off == 1 {
                let mut b0 = b[0];
                for i in 1..k {
                    b0 += c[i] * b[i];
                }
                b[0] = b0;
            }
            for (bi, ai) in b[off..].iter_mut().zip(&a[off..]) {
                *bi *= ai;
            }
        }
        for (p, u) in self.p.iter_mut().zip(self.ubuf.iter_mut()) {
            if factor {
                forms::factor_remap(p, u, a, c, off, true);
                continue;
            }
            for i in off..k {
                let ai = a[i];
                let row = &mut p[i * k..(i + 1) * k];
                if off == 1 {
                    row[0] = u[i] * ai;
                }
                for (pij, aj) in row[off..].iter_mut().zip(&a[off..]) {
                    *pij *= ai * aj;
                }
            }
            if off == 1 {
                for j in 1..k {
                    p[j] = u[j] * a[j];
                }
                p[0] = u[0];
            }
        }
        if let Some(s) = self.smoother.as_deref_mut() {
            s.remap(a, c, off);
        }
        true
    }

    /// Whether covariance `pi` is unsized: all zero on its diagonal, as it
    /// is from the start until a row sizes the noise (the module doc, CC4).
    /// A sized `P` has a positive diagonal. One that the reversion has
    /// shrunk to exactly 0, every slot reverting over hundreds of its
    /// half-lives with no process noise to restore it, holds nothing, and is
    /// sized again.
    fn is_unsized(&self, pi: usize) -> bool {
        if self.factor() {
            self.is_unsized_as::<true>(pi)
        } else {
            self.is_unsized_as::<false>(pi)
        }
    }

    /// [`Kalman::is_unsized`] in the form `FACTOR` names (`Kalman::sized`).
    fn is_unsized_as<const FACTOR: bool>(&self, pi: usize) -> bool {
        forms::unsized_of::<FACTOR>(&self.p[pi], self.cfg.k_total())
    }

    /// Whether target `j` is predicted, `min_weight` aside: it has weight,
    /// and its covariance is sized. An unsized prior's prediction is `z·b`
    /// at coefficients nothing has corrected, `0.0` whatever the target's
    /// level, and it would seed the residual variance with the level
    /// squared (review round 9, B-1; docs/PLAN.md task 270): standardized
    /// without `obs_var` the slots wait for three observed rows, which a
    /// weight column lets a row get past, and which the filter's own
    /// prediction, made whatever `min_weight` says, meets on every row.
    fn predicts(&self, j: usize) -> bool {
        if self.factor() {
            self.predicts_as::<true>(j)
        } else {
            self.predicts_as::<false>(j)
        }
    }

    /// [`Kalman::predicts`] in the form `FACTOR` names (`Kalman::sized`).
    fn predicts_as<const FACTOR: bool>(&self, j: usize) -> bool {
        let pi = if self.cfg.share_p { 0 } else { j };
        self.wj[j] > 0.0 && !self.is_unsized_as::<FACTOR>(pi)
    }

    /// Whether a target with weight waits for its prior to be sized, and so
    /// is not predicted ([`Self::predicts`]): the filter's own warm-up,
    /// the rows its noise estimate is taken from.
    pub fn sizing(&self) -> bool {
        (0..self.cfg.n_targets).any(|j| self.wj[j] > 0.0 && !self.predicts(j))
    }

    /// `share_p`'s noise before any target has a residual variance: the
    /// mean of the row's squared innovations over the targets it observes
    /// (present, at a positive weight), each read from `zb`, before any
    /// update. A square that is not finite gives no scale and is left out;
    /// 0 when nothing is left, or every innovation was exactly 0 (CC4).
    fn shared_first_noise(&self, y: &[Option<f64>], weight: f64) -> f64 {
        if weight > 0.0 {
            let squares = y.iter().zip(&self.zb).filter_map(|(yj, zb)| {
                let e = (*yj)? - zb;
                Some(e * e).filter(|e2| e2.is_finite())
            });
            let mut buf = Vec::new();
            let sum = sum_ascending(&mut buf, squares);
            if !buf.is_empty() {
                return first_noise(sum / buf.len() as f64);
            }
        }
        0.0
    }

    /// The transition's coupling in the anchor's coordinates, `g_i = c_i
    /// (phi_0 − phi_i)` for each slot that holds anything, into `g`, from
    /// the map `c`: `None` where it is 0 throughout -- without
    /// `standardize` or an intercept, or where every slot reverts at one
    /// half-life (the module doc).
    fn coupling(&self, phi: &[f64], c: &[f64], live: &[bool], g: &mut [f64]) -> bool {
        if !self.cfg.standardize || !self.cfg.fit_intercept || self.cfg.revert_half_life.len() == 1
        {
            return false;
        }
        let mut any = false;
        g[0] = 0.0;
        for i in 1..g.len() {
            g[i] = if live[i] {
                c[i] * (phi[0] - phi[i])
            } else {
                0.0
            };
            any |= g[i] != 0.0;
        }
        any
    }

    /// `b <- Phi_a b`, `P <- Phi_a P Phi_aᵀ` for a clock delta `d_clock`,
    /// `Phi_a` the transition in the anchor's coordinates (the module doc):
    /// the coefficient means shrink toward zero and the covariance with
    /// them. A no-op (and skipped) under the default random walk.
    fn transition<const FACTOR: bool>(&mut self, d_clock: f64) {
        if !self.cfg.reverts() {
            return;
        }
        if self.cfg.revert_half_life.len() == 1 {
            // One half-life for every slot: one exponential, not `k`.
            self.phi.fill(self.cfg.phi(0, d_clock));
        } else {
            for (i, ph) in self.phi.iter_mut().enumerate() {
                *ph = self.cfg.phi(i, d_clock);
            }
        }
        let mut g = std::mem::take(&mut self.gbuf);
        let coupled = self.coupling(&self.phi, &self.map_c, &self.live, &mut g);
        for b in &mut self.beta {
            transit(b, &self.phi, coupled.then_some(&g[..]));
        }
        for p in &mut self.p {
            forms::transit_cov::<FACTOR>(p, &self.phi, coupled.then_some(&g[..]), &mut self.vbuf);
        }
        if let Some(s) = self.smoother.as_deref_mut() {
            s.transit(&self.phi, coupled.then_some(&g[..]));
        }
        self.gbuf = g;
    }

    /// `Q D²` for the slots of covariance `pi` that are sized (a variance
    /// above 0), `D²` read on a reverting slot as its bounded form
    /// ([`KalmanCfg::gap_noise`]), defined in the current coordinates and
    /// carried into the anchor's (the module doc): an arrowhead, O(k).
    fn add_process_noise<const FACTOR: bool>(&mut self, pi: usize, sigma2: f64, d: f64) {
        if d == 0.0 {
            return;
        }
        if FACTOR {
            return self.factor_noise(pi, sigma2, d);
        }
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let mut q = std::mem::take(&mut self.qbuf);
        self.q_held_into(sigma2, &mut q);
        let open = self.open::<FACTOR>(pi);
        let gaps = self.gap_factors.of(&self.cfg, d);
        let p = &mut self.p[pi];
        let mut corner = 0.0;
        for i in 0..k {
            let qd = q[i] * gaps[i];
            if !self.cfg.standardize {
                // The whole `P` is sized at once (the caller saw to it), but
                // for a diffuse intercept, which its first row sizes.
                if !(open && i < off) {
                    p[i * k + i] += qd;
                }
                continue;
            }
            if p[i * k + i] == 0.0 || qd == 0.0 {
                continue;
            }
            if i < off {
                p[i * k + i] += qd;
                continue;
            }
            let a = self.map_a[i];
            if off == 1 {
                let v = self.map_c[i] / a;
                corner += v * v * qd;
                let o = -v * qd / a;
                p[i] += o;
                p[i * k] += o;
            }
            p[i * k + i] += qd / (a * a);
        }
        if corner != 0.0 {
            p[0] += corner;
        }
        self.qbuf = q;
    }

    /// Covariance `pi` as it stands after the last row: `P`, and after rows
    /// that observed nothing the noise of the clock since the last
    /// observation, `Q D²` on the sized slots, carried into the anchor's
    /// coordinates as the next observation will charge it (the module doc),
    /// at the noise the state holds (`obs_var`, else the residual variance;
    /// none before there is one). What `pred_var`, `se_coef` and the
    /// summary's readiness read: a coefficient's uncertainty grows across a
    /// gap in its target, as it did when the noise was charged per row.
    fn covariance_now(&self, pi: usize) -> std::borrow::Cow<'_, [f64]> {
        let d = self.elapsed[pi];
        let noise = if self.cfg.obs.is_gaussian_identity() {
            self.readiness_noise(pi)
        } else {
            1.0
        };
        // `readiness_noise` is a positive number, or NaN for none.
        if d == 0.0 || noise.is_nan() || self.is_unsized(pi) {
            return self.cov(pi);
        }
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let mut q = vec![0.0; k];
        self.q_held_into(noise, &mut q);
        let live: Vec<bool> = (0..k).map(|i| self.is_live(i)).collect();
        let (mut a, mut c) = (vec![1.0; k], vec![0.0; k]);
        self.map_into(&live, &mut a, &mut c);
        // A diffuse intercept waiting for its first row takes no noise, as
        // `add_process_noise` keeps it off (review 2026-11-23, KAL-5).
        let open = if self.factor() {
            self.open::<true>(pi)
        } else {
            self.open::<false>(pi)
        };
        let mut p = self.cov(pi).into_owned();
        for i in 0..k {
            let gn = if self.cfg.obs.is_linear() {
                self.cfg.gap_noise(i, d)
            } else {
                nonlinear::gap_noise_portable(&self.cfg, i, d)
            };
            let qd = q[i] * gn;
            if qd == 0.0 || (self.cfg.standardize && p[i * k + i] == 0.0) || (open && i < off) {
                continue;
            }
            if !self.cfg.standardize || i < off {
                p[i * k + i] += qd;
                continue;
            }
            if off == 1 {
                let v = c[i] / a[i];
                p[0] += v * v * qd;
                let o = -v * qd / a[i];
                p[i] += o;
                p[i * k] += o;
            }
            p[i * k + i] += qd / (a[i] * a[i]);
        }
        std::borrow::Cow::Owned(p)
    }

    /// The prior `p0 R` of each slot of covariance `pi` that is unsized and
    /// whose scale is usable, defined in the current coordinates and
    /// carried into the anchor's (the module doc). A slot that holds nothing
    /// anywhere has its anchor set to the moments as they stand first, so it
    /// takes its prior as it is. The intercept's too, unless `intercept` is
    /// false: in units of the noise it is left diffuse for the first row
    /// that observes its target (the module doc).
    fn size_slots(&mut self, pi: usize, noise: f64, intercept: bool) {
        // The form read once for the loop over the slots (`mark_live`).
        if self.factor() {
            self.size_slots_as::<true>(pi, noise, intercept, false);
        } else {
            self.size_slots_as::<false>(pi, noise, intercept, false);
        }
    }

    fn size_slots_as<const FACTOR: bool>(
        &mut self,
        pi: usize,
        noise: f64,
        intercept: bool,
        seed_prior: bool,
    ) -> bool {
        let v = self.cfg.p0 * noise;
        if !(v > 0.0 && v.is_finite()) {
            return false;
        }
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let mut seeded = false;
        for i in 0..k {
            if self.sized::<FACTOR>(pi, i) || !self.current_scale(i).1 {
                continue;
            }
            if i < off {
                if intercept {
                    self.add_slot_as::<FACTOR>(pi, 0, v, 1.0, None);
                    self.live[0] = true;
                }
                continue;
            }
            if !self.live[i] {
                self.anchor_slot_here(i);
            }
            let a = self.map_a[i];
            let open = self.open::<FACTOR>(pi);
            let cc = (off == 1 && !open).then(|| self.map_c[i] / a);
            self.add_slot_as::<FACTOR>(pi, i, v, a, cc);
            self.live[i] = true;
            if seed_prior && self.seed_standardized_slot_prior(pi, i, off, open) {
                seeded = true;
            }
        }
        seeded
    }

    /// Whether covariance `pi` has lost its positive-definiteness where it
    /// shows, on the diagonal: a variance below 0, or not a finite number;
    /// or a variance of exactly 0 beside a covariance that is not, since
    /// `|P_ij|² ≤ P_ii P_jj` holds of every positive semi-definite matrix.
    /// An update at the input bound can round a slot's variance to 0 and
    /// leave its row (docs/PLAN.md task 278, b32), and such a `P` read as
    /// unsized before the transition and sized after it, whose coupling
    /// writes the intercept's variance from those covariances: `predict`
    /// withheld what `step` predicted. O(k), and O(k²) only on the rows a
    /// variance is 0, which a row can afford where an eigensolve is not.
    fn lost_definiteness<const FACTOR: bool>(&self, pi: usize) -> bool {
        let k = self.cfg.k_total();
        let p = &self.p[pi];
        if FACTOR {
            // A factor's `C Cᵀ` is positive semi-definite while it is a
            // set of numbers.
            return !p.iter().all(|v| v.is_finite());
        }
        (0..k).any(|i| {
            let v = p[i * k + i];
            !(v >= 0.0 && v.is_finite())
                || (v == 0.0 && p[i * k..(i + 1) * k].iter().any(|c| *c != 0.0))
        })
    }

    /// Covariance `pi` emptied and sized again from the noise as the state
    /// holds it (`obs_var`, else the residual variance), as at the start:
    /// `p0` times it on each slot whose scale is usable, carried into the
    /// anchor's coordinates, or the whole diagonal without `standardize`.
    /// With no noise yet it stays unsized, and the next row that observes
    /// its target sizes it as the first did.
    fn size_again(&mut self, pi: usize) {
        if let Some(s) = self.smoother.as_deref_mut() {
            s.cut(Some(pi));
        }
        self.p[pi].fill(0.0);
        if self.units() || !self.cfg.obs.is_gaussian_identity() {
            // In units of the noise the prior needs none: `p0` on every
            // usable slot, the intercept's too, its level already in `b`.
            if self.cfg.standardize {
                self.size_slots(pi, 1.0, true);
            } else {
                self.size_whole(pi, self.cfg.p0);
            }
            return;
        }
        let j = if self.cfg.share_p { 0 } else { pi };
        let r = self.readiness_noise(j);
        if r.is_nan() {
            return;
        }
        if self.cfg.standardize {
            self.size_slots(pi, r, true);
        } else {
            self.size_whole(pi, self.cfg.p0 * r);
        }
    }

    /// Unstandardized, covariance `pi` sized whole to `v I`, where `v` is a
    /// positive number: the first row with a noise does it, at `p0` times
    /// that noise, and so does [`Self::size_again`].
    fn size_whole(&mut self, pi: usize, v: f64) {
        let k = self.cfg.k_total();
        if v > 0.0 && v.is_finite() {
            let v = if self.form == CovarianceForm::SquareRoot {
                v.sqrt()
            } else {
                v
            };
            let p = &mut self.p[pi];
            p.fill(0.0);
            for i in 0..k {
                p[i * k + i] = v;
            }
        }
    }

    /// The noise target `j`'s readiness is read against, as the state holds
    /// it before a row (the module doc): `obs_var`, else the residual
    /// variance -- under `share_p` the mean over the targets that have one,
    /// summed in ascending order as `step` sums it -- and NaN, none, before
    /// there is one. Never the row's own innovation, which reads its target.
    fn readiness_noise(&self, j: usize) -> f64 {
        if let Some(v) = self.cfg.obs_var {
            return v;
        }
        let s2 = if self.cfg.share_p {
            shared_noise(&mut Vec::new(), &self.sig2)
        } else {
            self.sig2[j]
        };
        if s2 > 0.0 && s2.is_finite() {
            s2
        } else {
            f64::NAN
        }
    }

    /// `z' P⁻ z` against covariance `pi` for the row `x`, `d_clock` after the
    /// last row, at the noise `r` the process noise is derived from: `P`
    /// carried through the transition and the process noise the
    /// observation would add for the clock since the last one, `D + d`,
    /// without moving it -- `wᵀ P w + (D + d)² Σ_i q_i z_i²` over the sized
    /// slots, with `w = Phi_aᵀ z^a` and `z` the row in the current
    /// coordinates, which is `step`'s `P <- Phi_a P Phi_aᵀ + Q_a D²` read
    /// through `z^a`.
    fn prior_quad(&self, pi: usize, x: &[f64], d_clock: f64, r: f64) -> f64 {
        let k = self.cfg.k_total();
        let p = self.cov(pi);
        let p = &p[..];
        let za = self.anchored(x);
        let w: Vec<f64> = if self.cfg.reverts() {
            let phi: Vec<f64> = (0..k).map(|i| self.cfg.phi(i, d_clock)).collect();
            let live: Vec<bool> = (0..k).map(|i| self.is_live(i)).collect();
            let (mut a, mut c, mut g) = (vec![1.0; k], vec![0.0; k], vec![0.0; k]);
            self.map_into(&live, &mut a, &mut c);
            let coupled = self.coupling(&phi, &c, &live, &mut g);
            // w = Phi_aᵀ z^a: w_0 = phi_0 z_0, w_i = g_i z_0 + phi_i z_i.
            (0..k)
                .map(|i| {
                    if coupled && i > 0 {
                        g[i] * za[0] + phi[i] * za[i]
                    } else {
                        phi[i] * za[i]
                    }
                })
                .collect()
        } else {
            za
        };
        let mut quad = 0.0;
        for i in 0..k {
            let row = i * k;
            let mut acc = 0.0;
            for jj in 0..k {
                acc += p[row + jj] * w[jj];
            }
            quad += w[i] * acc;
        }
        let z = self.standardized(x);
        let mut q = vec![0.0; k];
        self.q_held_into(r, &mut q);
        let gap = self.elapsed[pi] + d_clock;
        let dd = gap.powi(2);
        let sized = |i: &usize| p[i * k + i] != 0.0;
        let walking = |i: &usize| self.cfg.revert_of(*i).is_infinite();
        let noise: f64 = (0..k)
            .filter(sized)
            .filter(walking)
            .map(|i| q[i] * z[i] * z[i])
            .sum();
        // A reverting slot's noise is bounded in the gap (`gap_noise`), so
        // it is summed apart; with none, the sum is the random walk's alone,
        // to the bit.
        let reverting: f64 = (0..k)
            .filter(sized)
            .filter(|i| !walking(i))
            .map(|i| q[i] * self.cfg.gap_noise(i, gap) * z[i] * z[i])
            .sum();
        // A rounding below 0 is 0, as in `current_diagonal` (KAL-7).
        crate::solve::clamp_rounding(quad) + dd * noise + reverting
    }

    /// `sqrt(1 + z' P⁻ z / R)` per target for the row `x` at `d_clock`
    /// (the module doc): infinite while `P` is unsized or the noise is
    /// none, NaN where a feature is not a number.
    fn row_inflation_into(&self, x: &[f64], d_clock: f64, out: &mut Vec<f64>) {
        let m = self.cfg.n_targets;
        out.clear();
        out.resize(m, f64::INFINITY);
        for (j, o) in out.iter_mut().enumerate() {
            let pi = if self.cfg.share_p { 0 } else { j };
            let r = self.readiness_noise(j);
            if self.is_unsized(pi) || r.is_nan() {
                continue;
            }
            *o = (1.0 + self.per_noise(self.prior_quad(pi, x, d_clock, r), r)).sqrt();
        }
    }

    /// `E[z_i²]` per slot over the rows the standardizer has seen, as `z`
    /// is formed from them in the current coordinates: 1 for the intercept,
    /// `var / s²` centred and `raw / s²` through the origin, the raw second
    /// moment unstandardized.
    fn design_second_moments(&self) -> Vec<f64> {
        let off = usize::from(self.cfg.fit_intercept);
        let s = self.scales();
        (0..self.cfg.k_total())
            .map(|i| {
                if i < off {
                    1.0
                } else if !self.cfg.standardize {
                    self.stats.raw(i)
                } else if off == 0 {
                    self.stats.raw(i) / (s[i] * s[i])
                } else {
                    self.stats.var(i) / (s[i] * s[i])
                }
            })
            .collect()
    }

    /// The diagonal of covariance `pi` as it stands
    /// ([`Self::covariance_now`]) in the current coordinates, `Mᵀ P M` (the
    /// module doc): `a_i² P_ii`, and with an intercept `u_0 + Σ_j c_j u_j`
    /// with `u = P_0· + Σ_l c_l P_l·`.
    fn current_diagonal(&self, pi: usize) -> Vec<f64> {
        let k = self.cfg.k_total();
        let p = self.covariance_now(pi);
        let p = &p[..];
        if !self.cfg.standardize {
            return (0..k).map(|i| p[i * k + i]).collect();
        }
        let off = usize::from(self.cfg.fit_intercept);
        let live: Vec<bool> = (0..k).map(|i| self.is_live(i)).collect();
        let (mut a, mut c) = (vec![1.0; k], vec![0.0; k]);
        self.map_into(&live, &mut a, &mut c);
        let mut out: Vec<f64> = (0..k).map(|i| a[i] * a[i] * p[i * k + i]).collect();
        if off == 1 {
            let mut u = p[0..k].to_vec();
            for l in 1..k {
                for (uj, plj) in u.iter_mut().zip(&p[l * k..(l + 1) * k]) {
                    *uj += c[l] * plj;
                }
            }
            // `eᵀ P e`, `e = (1, c)`, cancels where `c` is past the re-map's
            // limit: a rounding below 0 is 0 (review 2026-11-23, KAL-7).
            out[0] = crate::solve::clamp_rounding(u[0] + (1..k).map(|j| c[j] * u[j]).sum::<f64>());
        }
        out
    }

    /// `T P Tᵀ`'s diagonal for covariance `pi` as it stands
    /// ([`Self::covariance_now`]): the variance of each coefficient as
    /// `coefficients` reads it out (the module doc).
    fn coef_variance_of(&self, pi: usize) -> Vec<f64> {
        let k = self.cfg.k_total();
        let p = self.covariance_now(pi);
        let p = &p[..];
        if !self.cfg.standardize {
            return (0..k).map(|i| p[i * k + i]).collect();
        }
        let off = usize::from(self.cfg.fit_intercept);
        let s = &self.anchor_scale;
        let mut out: Vec<f64> = (0..k)
            .map(|i| {
                if i < off {
                    p[i * k + i]
                } else {
                    p[i * k + i] / (s[i] * s[i])
                }
            })
            .collect();
        if off == 1 {
            // c_0 = b_0 − Σ_i b_i m^a_i / s^a_i: v' P v with v = [1, −m^a / s^a].
            let v: Vec<f64> = (0..k)
                .map(|i| {
                    if i == 0 {
                        1.0
                    } else {
                        -(self.anchor_hi[i] + self.anchor_lo[i]) / s[i]
                    }
                })
                .collect();
            let mut quad = 0.0;
            for i in 0..k {
                let mut acc = 0.0;
                for jj in 0..k {
                    acc += p[i * k + jj] * v[jj];
                }
                quad += v[i] * acc;
            }
            // A rounding below 0 is 0; a NaN stays NaN, where `f64::max`
            // read it as 0.0, an exact intercept (review round 9, B-2; the
            // sandwich's twin). Not reached by a test: a NaN here needs a
            // NaN in `P`, which `size_again` refuses at the end of the row
            // that made it, or `±inf` terms cancelling in `v' P v`, which
            // needs `P` near `f64`'s largest number (`p0` times a noise
            // near 1e300) beside a feature far from its scale.
            out[0] = crate::solve::clamp_rounding(quad);
        }
        out
    }

    /// Under `share_p`, target `j`'s covariance is the shared one times
    /// `σ²_j / σ̄²` (the module doc, task 211): that ratio, 1 with `obs_var`,
    /// unshared, or while the target has no residual variance.
    /// In units of the noise, `P` is scaled by target `j`'s own noise, shared
    /// or not, and NaN until it has one.
    fn own_noise_ratio(&self, j: usize) -> f64 {
        if self.units() {
            let own = self.sig2[j];
            return if own > 0.0 { own } else { f64::NAN };
        }
        if !self.cfg.share_p || self.cfg.obs_var.is_some() {
            return 1.0;
        }
        let mean = shared_noise(&mut Vec::new(), &self.sig2);
        let own = self.sig2[j];
        if own > 0.0 && mean > 0.0 {
            own / mean
        } else {
            1.0
        }
    }
}

/// `b <- Phi_a b` in place (the module doc): `b_i <- phi_i b_i`, and with
/// the coupling `g`, `b_0 <- phi_0 b_0 + Σ_i g_i b_i` from the `b_i` before
/// the row. `step` and `predict` both run it, so the two agree to the bit.
fn transit(b: &mut [f64], phi: &[f64], g: Option<&[f64]>) {
    if let Some(g) = g {
        let mut b0 = b[0] * phi[0];
        for i in 1..b.len() {
            b0 += g[i] * b[i];
        }
        for (bi, ph) in b.iter_mut().zip(phi).skip(1) {
            *bi *= ph;
        }
        b[0] = b0;
    } else {
        for (bi, ph) in b.iter_mut().zip(phi) {
            *bi *= ph;
        }
    }
}

/// How far one re-map may move the coordinates for `b` and `P` to follow
/// (the module doc): a scale by this factor either way, a mean by this many
/// of the anchor's scale. `Mᵀ P M` puts `c² P_ii` into the intercept's
/// variance and reads it back through a cancellation on every later row, so
/// a shift of `c` costs `c²` rounding steps of `P`'s narrowest direction:
/// 2.3e-10 of it at 1024, and all of it at the input bound's 1e99, where the
/// re-map left a slope's variance at −3.6e149 and the filter never
/// recovered (`tests/model_contract.rs`, `kalman_recovers_from_bounded_extremes`,
/// the reverting filter). A row of ordinary data moves a mean by a fraction
/// of a scale: past this it is thousands of standard deviations out.
const REMAP_LIMIT: f64 = 1024.0;

/// A prediction, or none (NaN) when it is not a number. A feature at the
/// input bound, standardized against a scale the earlier rows set, times its
/// coefficient can overflow `z . beta` to `inf`, and an infinite prediction
/// is no forecast: `step` and `predict` both withhold it, as `step` already
/// skips the update such a row would poison (docs/IMPROVEMENTS.md C2). Found
/// by the generated stream in `tests/model_contract.rs` (docs/PLAN.md
/// task 158).
/// The clock since any target was last observed: the least of each
/// covariance's ([`Kalman`]'s `elapsed`).
fn unobserved(elapsed: &[f64]) -> f64 {
    elapsed.iter().copied().fold(f64::INFINITY, f64::min)
}

fn a_number_or_none(v: f64) -> f64 {
    if v.is_finite() { v } else { f64::NAN }
}

/// A squared innovation, or a mean of them, as the noise a target takes
/// before it has a residual variance (the module doc, CC4): itself, or 0 --
/// no noise -- where it gives no scale. That is an innovation of exactly 0,
/// or one whose square underflows to 0, and a square that is not finite.
fn first_noise(e2: f64) -> f64 {
    if e2.is_finite() { e2 } else { 0.0 }
}

/// The sum of `vals` in ascending order, through `buf`: the same bits
/// whatever order the targets come in, as `share_p`'s mean noise must be
/// (the module doc).
fn sum_ascending(buf: &mut Vec<f64>, vals: impl Iterator<Item = f64>) -> f64 {
    buf.clear();
    buf.extend(vals);
    buf.sort_unstable_by(f64::total_cmp);
    buf.iter().sum()
}

/// `share_p`'s noise once a target has a residual variance: the mean `σ²`
/// over the targets that have one (`σ² > 0`), summed in ascending order
/// through `buf`; 0, no noise yet, where none has (the module doc). A target
/// with no residual variance has no estimate of the noise, not a noise of 0:
/// counted in the mean, a target null so far halved the noise its neighbour
/// was weighed against (review round 5, A1). `shared_first_noise` averages
/// over the targets the row observes the same way.
fn shared_noise(buf: &mut Vec<f64>, sig2: &[f64]) -> f64 {
    let sum = sum_ascending(buf, sig2.iter().copied().filter(|s2| *s2 > 0.0));
    if buf.is_empty() {
        0.0
    } else {
        sum / buf.len() as f64
    }
}

/// The exact step of a prior diffuse in the intercept (Durbin and Koopman,
/// *Time Series Analysis by State Space Methods*, §5.2), for a covariance
/// whose intercept's row and column are 0: `P` the prior of the other slots,
/// `pz = P z` (its intercept's entry 0, as `z_0 = 1`) and `s = z' P z + 1 /
/// w`. The intercept's variance becomes `s` and its covariance with slot
/// `i` `−(P z)_i`; the other slots' block is unchanged. That is the
/// posterior of `b_0 = y − z_{1..}' b_{1..}` given the row, which the
/// limit of `P_00 → ∞` in the ordinary update reaches; its gain is `e_0`.
fn diffuse_step(p: &mut [f64], pz: &[f64], s: f64) {
    let k = pz.len();
    p[0] = s;
    for i in 1..k {
        p[i] = -pz[i];
        p[i * k] = -pz[i];
    }
}

/// `P <- P - g (P z)ᵀ`, once per pair and written to both halves, so `P`
/// stays symmetric (see `Rls::step` for why that matters). Pair `(i, j)` with
/// `i <= j` takes `g_i (P z)_j` at `(i, j)` and at its mirror `(j, i)`, so
/// each row of `P` is walked along its own memory: row `r` takes `g_c pz_r`
/// at the columns `c` before the diagonal (the mirrors) and `g_r pz_c` from
/// the diagonal on. The same subtraction of the same product from each
/// entry as walking the upper triangle and mirroring each, with no strided
/// access.
fn take_the_row(p: &mut [f64], gain: &[f64], pz: &[f64]) {
    let k = gain.len();
    if k == 0 {
        return;
    }
    for (i, row) in p.chunks_exact_mut(k).enumerate() {
        let (below, from) = row.split_at_mut(i);
        let pi = pz[i];
        for (pij, gj) in below.iter_mut().zip(gain) {
            *pij -= gj * pi;
        }
        let gi = gain[i];
        for (pij, pj) in from.iter_mut().zip(&pz[i..]) {
            *pij -= gi * pj;
        }
    }
}

/// `pz = P z`: each entry the row's products summed in column order from 0,
/// four rows at a time so four chains of adds overlap.
fn matvec(p: &[f64], z: &[f64], out: &mut [f64]) {
    let k = z.len();
    if k == 0 {
        return;
    }
    let mut i = 0;
    while i + 4 <= out.len() {
        let (r0, rest) = p[i * k..(i + 4) * k].split_at(k);
        let (r1, rest) = rest.split_at(k);
        let (r2, r3) = rest.split_at(k);
        let (mut a0, mut a1, mut a2, mut a3) = (0.0, 0.0, 0.0, 0.0);
        for j in 0..k {
            let zj = z[j];
            a0 += r0[j] * zj;
            a1 += r1[j] * zj;
            a2 += r2[j] * zj;
            a3 += r3[j] * zj;
        }
        out[i] = a0;
        out[i + 1] = a1;
        out[i + 2] = a2;
        out[i + 3] = a3;
        i += 4;
    }
    for (row, o) in p[i * k..].chunks_exact(k).zip(&mut out[i..]) {
        let mut acc = 0.0;
        for (pv, zj) in row.iter().zip(z) {
            acc += pv * zj;
        }
        *o = acc;
    }
}

impl OnlineModel for Kalman {
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend_from_slice(&self.wj);
        true
    }

    /// Without a row: the per-row statistic's mean field over the design at
    /// the posterior, `sqrt(1 + Σ_i P_ii E[z_i²] / R)` in the current
    /// coordinates (the module doc).
    fn error_inflation_into(&self, out: &mut Vec<f64>) -> bool {
        if !self.cfg.obs.is_gaussian_identity() {
            return false;
        }
        let m = self.cfg.n_targets;
        out.clear();
        out.resize(m, f64::INFINITY);
        let ez2 = self.design_second_moments();
        for (j, o) in out.iter_mut().enumerate() {
            let pi = if self.cfg.share_p { 0 } else { j };
            let r = self.readiness_noise(j);
            if self.is_unsized(pi) || r.is_nan() {
                continue;
            }
            let diag = self.current_diagonal(pi);
            let trace: f64 = diag.iter().zip(&ez2).map(|(p, e)| p * e).sum();
            *o = (1.0 + self.per_noise(trace, r)).sqrt();
        }
        true
    }

    /// The gate reads the row's own `sqrt(1 + z' P⁻ z / R)`: exact, `O(k²)`.
    fn error_inflation_gate_into(
        &self,
        x: &[f64],
        d_clock: f64,
        out: &mut Vec<f64>,
        _limit: f64,
    ) -> bool {
        if !self.cfg.obs.is_gaussian_identity() {
            return false;
        }
        self.row_inflation_into(x, d_clock, out);
        true
    }

    fn row_error_inflation_into(&self, x: &[f64], d_clock: f64, out: &mut Vec<f64>) -> bool {
        if !self.cfg.obs.is_gaussian_identity() {
            return false;
        }
        self.row_inflation_into(x, d_clock, out);
        true
    }

    fn row_gram_leverage_into(
        &self,
        x: &[f64],
        d_clock: f64,
        need_vec: bool,
        h_unit: &mut Vec<f64>,
        g_inv_x: &mut Vec<f64>,
        diag_g_inv: &mut Vec<f64>,
    ) -> bool {
        let m = self.cfg.n_targets;
        let k = self.cfg.k_total();
        h_unit.clear();
        h_unit.resize(m, f64::NAN);
        if need_vec {
            g_inv_x.clear();
            g_inv_x.resize(m * k, f64::NAN);
            diag_g_inv.clear();
            diag_g_inv.resize(m * k, f64::NAN);
        }
        let off = usize::from(self.cfg.fit_intercept);
        for (j, h_slot) in h_unit.iter_mut().enumerate().take(m) {
            let pi = if self.cfg.share_p { 0 } else { j };
            let r = self.readiness_noise(j);
            if self.is_unsized(pi) || r.is_nan() {
                continue;
            }
            let h = self.per_noise(self.prior_quad(pi, x, d_clock, r), r);
            if h.is_finite() {
                *h_slot = h;
            }
            if !need_vec {
                continue;
            }
            let p_raw = self.cov(pi);
            let p_raw = &p_raw[..];
            let za = self.anchored(x);
            // Prior covariance P_a = Phi_a P Phi_a^T in anchored coords, and
            // u_a = P_a * za.
            let (mut pa, w_vec) = if self.cfg.reverts() {
                let phi: Vec<f64> = (0..k).map(|i| self.cfg.phi(i, d_clock)).collect();
                let live: Vec<bool> = (0..k).map(|i| self.is_live(i)).collect();
                let (mut a_m, mut c_m, mut g_m) = (vec![1.0; k], vec![0.0; k], vec![0.0; k]);
                self.map_into(&live, &mut a_m, &mut c_m);
                let coupled = self.coupling(&phi, &c_m, &live, &mut g_m);
                // Apply Phi_a from left and right to P: Phi_a has diagonal `phi`
                // and optional row-0 coupling `g_m[1..k]`.
                let mut tmp = vec![0.0; k * k];
                for r_i in 0..k {
                    for c_i in 0..k {
                        let mut v = phi[r_i] * p_raw[r_i * k + c_i];
                        if coupled && r_i == 0 {
                            for l in 1..k {
                                v += g_m[l] * p_raw[l * k + c_i];
                            }
                        }
                        tmp[r_i * k + c_i] = v;
                    }
                }
                let mut out_p = vec![0.0; k * k];
                for r_i in 0..k {
                    for c_i in 0..k {
                        let mut v = tmp[r_i * k + c_i] * phi[c_i];
                        if coupled && c_i == 0 {
                            for l in 1..k {
                                v += tmp[r_i * k + l] * g_m[l];
                            }
                        }
                        out_p[r_i * k + c_i] = v;
                    }
                }
                (out_p, za)
            } else {
                (p_raw.to_vec(), za)
            };
            let _ = &mut pa;
            let mut u_a = vec![0.0; k];
            for i in 0..k {
                let mut acc = 0.0;
                for c_i in 0..k {
                    acc += pa[i * k + c_i] * w_vec[c_i];
                }
                u_a[i] = acc;
            }
            // Process noise effective diagonal in current standardized coords.
            let z_cur = self.standardized(x);
            let mut q_h = vec![0.0; k];
            self.q_held_into(r, &mut q_h);
            let gap = self.elapsed[pi] + d_clock;
            let dd = gap * gap;
            let mut q_eff = vec![0.0; k];
            for i in 0..k {
                if p_raw[i * k + i] != 0.0 {
                    q_eff[i] = if self.cfg.revert_of(i).is_infinite() {
                        dd * q_h[i]
                    } else {
                        q_h[i] * self.cfg.gap_noise(i, gap)
                    };
                }
            }
            // Map both anchored P_a and current Q_eff to original caller units.
            let slot_off = j * k;
            if !self.cfg.standardize {
                for i in 0..k {
                    g_inv_x[slot_off + i] = self.per_noise(u_a[i] + q_eff[i] * z_cur[i], r);
                    diag_g_inv[slot_off + i] = self.per_noise(pa[i * k + i] + q_eff[i], r);
                }
            } else {
                let s_a = &self.anchor_scale;
                let s_cur = self.scales();
                let (m_cur, m_cur_lo, _) = self.stats.parts();
                // T_a * u_a + T_cur * (q_eff .* z_cur)
                for i in 0..k {
                    if i < off {
                        let mut v_a = u_a[0];
                        let mut v_q = q_eff[0] * z_cur[0];
                        for l in 1..k {
                            let ma = self.anchor_hi[l] + self.anchor_lo[l];
                            v_a -= (ma / s_a[l]) * u_a[l];
                            let mc = m_cur[l] + m_cur_lo[l];
                            v_q -= (mc / s_cur[l]) * q_eff[l] * z_cur[l];
                        }
                        g_inv_x[slot_off] = self.per_noise(v_a + v_q, r);
                    } else {
                        g_inv_x[slot_off + i] =
                            self.per_noise(u_a[i] / s_a[i] + q_eff[i] * z_cur[i] / s_cur[i], r);
                    }
                }
                // diag(T_a P_a T_a^T + T_cur diag(q_eff) T_cur^T)
                for i in off..k {
                    let da = pa[i * k + i] / (s_a[i] * s_a[i]);
                    let dq = q_eff[i] / (s_cur[i] * s_cur[i]);
                    diag_g_inv[slot_off + i] = self.per_noise(da + dq, r);
                }
                if off == 1 {
                    let va: Vec<f64> = (0..k)
                        .map(|i| {
                            if i == 0 {
                                1.0
                            } else {
                                -(self.anchor_hi[i] + self.anchor_lo[i]) / s_a[i]
                            }
                        })
                        .collect();
                    let mut quad_a = 0.0;
                    for r_i in 0..k {
                        let mut acc = 0.0;
                        for c_i in 0..k {
                            acc += pa[r_i * k + c_i] * va[c_i];
                        }
                        quad_a += va[r_i] * acc;
                    }
                    let mut quad_q = q_eff[0];
                    for l in 1..k {
                        let mc = m_cur[l] + m_cur_lo[l];
                        let ratio = mc / s_cur[l];
                        quad_q += q_eff[l] * ratio * ratio;
                    }
                    diag_g_inv[slot_off] =
                        self.per_noise(crate::solve::clamp_rounding(quad_a) + quad_q, r);
                }
            }
        }
        true
    }

    /// `T P Tᵀ`'s diagonal per target, absolute: `P` carries the noise, and
    /// under `share_p` target `j`'s is the shared one times `σ²_j / σ̄²`
    /// (the module doc). NaN for a target whose `P` is unsized.
    fn coef_variance(&self) -> Option<crate::CoefVariance> {
        let (m, k) = (self.cfg.n_targets, self.cfg.k_total());
        let per = (0..m)
            .map(|j| {
                let pi = if self.cfg.share_p { 0 } else { j };
                if self.is_unsized(pi) {
                    vec![f64::NAN; k]
                } else {
                    let ratio = self.own_noise_ratio(j);
                    self.coef_variance_of(pi)
                        .into_iter()
                        .map(|v| v * ratio)
                        .collect()
                }
            })
            .collect();
        Some(crate::CoefVariance::Absolute(per))
    }

    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        if !self.cfg.obs.is_linear() {
            self.step_nonlinear(x, y, d_clock, weight, out);
            return;
        }
        // The form read once a row, and the row stepped in it: each loop
        // over the slots in the covariance form reads the one entry it did
        // before the factor was added, at the cost it had then.
        if self.factor() {
            self.step_as::<true>(x, y, d_clock, weight, out);
        } else {
            self.step_as::<false>(x, y, d_clock, weight, out);
        }
    }

    /// The prediction at another `min_weight` floor, set for the read and
    /// put back ([`OnlineModel::predict_with_floor`]).
    fn predict_with_floor(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        floor: f64,
    ) -> Step {
        let floor = std::mem::replace(&mut self.cfg.min_weight, floor);
        let out = OnlineModel::predict_with(self, x, y, d_clock);
        self.cfg.min_weight = floor;
        out
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if !self.cfg.obs.is_linear() {
            return self.predict_nonlinear(x, d_clock);
        }
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let m = self.cfg.n_targets;
        let k = self.cfg.k_total();
        let n_eff = self.stats.n_eff();
        let mut pred = vec![f64::NAN; m];
        if n_eff >= self.cfg.min_weight {
            let zs = self.anchored(x);
            // The same numbers `step` would emit: its transition moves `b`
            // by `transit` before the dot product.
            let reverts = self.cfg.reverts();
            let (phi, g, coupled) = if reverts {
                let phi: Vec<f64> = (0..k).map(|i| self.cfg.phi(i, d_clock)).collect();
                let live: Vec<bool> = (0..k).map(|i| self.is_live(i)).collect();
                let (mut a, mut c, mut g) = (vec![1.0; k], vec![0.0; k], vec![0.0; k]);
                self.map_into(&live, &mut a, &mut c);
                let coupled = self.coupling(&phi, &c, &live, &mut g);
                (phi, g, coupled)
            } else {
                (vec![], vec![], false)
            };
            // Whether each covariance is sized as the step will read it:
            // after the transition, which across a long enough reversion gap
            // takes every entry to 0 (review fable2, C6). Before it, `predict`
            // gave `z·b` at a `b` shrunk to nothing where the step withheld.
            let sized_after: Vec<bool> = if reverts {
                let g = coupled.then_some(&g[..]);
                let mut v = vec![0.0; k];
                (0..self.p.len())
                    .map(|pi| {
                        let mut p = self.p[pi].clone();
                        if self.factor() {
                            forms::transit_cov::<true>(&mut p, &phi, g, &mut v);
                            !forms::unsized_of::<true>(&p, k)
                        } else {
                            forms::transit_cov::<false>(&mut p, &phi, g, &mut v);
                            !forms::unsized_of::<false>(&p, k)
                        }
                    })
                    .collect()
            } else {
                vec![]
            };
            for (j, p) in pred.iter_mut().enumerate() {
                let sized = if reverts {
                    self.wj[j] > 0.0 && sized_after[if self.cfg.share_p { 0 } else { j }]
                } else {
                    self.predicts(j)
                };
                if sized {
                    let v = if reverts {
                        let mut b = self.beta[j].clone();
                        transit(&mut b, &phi, coupled.then_some(&g[..]));
                        zs.iter().zip(&b).map(|(z, b)| z * b).sum()
                    } else {
                        zs.iter().zip(&self.beta[j]).map(|(z, b)| z * b).sum()
                    };
                    *p = a_number_or_none(v);
                }
            }
            // The forecasts at the horizons, from the state `step` reads.
            if !self.cfg.horizons.is_empty() {
                let g = coupled.then_some(&g[..]);
                let ahead = self.ahead_of_the_read(&pred, &zs, reverts, &phi, g, d_clock);
                pred.extend_from_slice(&ahead);
            }
        }
        let per = 3 * self.cfg.horizons.len();
        pred.resize(m * (1 + per), f64::NAN);
        let extra = self
            .cfg
            .emits_innovation()
            .then(|| crate::model::Extra::Kalman {
                values: vec![f64::NAN; self.cfg.extra_width()],
            });
        Step { pred, n_eff, extra }
    }

    fn predict_with(&self, x: &[f64], y: &[Option<f64>], d_clock: f64) -> Step {
        self.predict_with_weight(x, y, d_clock, 1.0)
    }

    fn predict_with_into(&self, x: &[f64], y: &[Option<f64>], d_clock: f64, out: &mut Step) {
        self.predict_with_weight_into(x, y, d_clock, 1.0, out);
    }

    fn label_next(&mut self, label: crate::ClockValue) {
        self.smooth_label(label);
    }

    fn hold_smoothed(&mut self, target: usize) {
        self.smooth_hold(target);
    }

    fn smoothed_into(&self, out: &mut Smoothed) -> bool {
        self.smoothed().is_some_and(|s| {
            out.clone_from(s);
            true
        })
    }

    fn state(&self) -> State {
        State::new(ModelState::Kalman(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Kalman(m) => {
                let mut m = (**m).clone();
                // The shapes are checked as the state is read (`KalmanV4`).
                crate::model::check_cfg("kalman", m.cfg.validate())?;
                m.ensure_buffers();
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "kalman",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    /// A prediction a target, then each target's forecasts at the horizons,
    /// three a horizon.
    fn n_outputs(&self) -> usize {
        self.cfg.n_targets * (1 + 3 * self.cfg.horizons.len())
    }
}

impl Kalman {
    /// [`OnlineModel::step_into`] in the covariance form `FACTOR` names
    /// (`Kalman::sized`).
    fn step_as<const FACTOR: bool>(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        // A value that is not usable, by the rule every model keeps
        // (`OnlineModel`): a feature that is not a number reached the
        // standardiser's moments (docs/PLAN.md task 183).
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        // Here, not before the refusal, which steps the row again.
        if self.smoother.is_some() {
            self.smooth_begin(d_clock);
        }
        self.ensure_buffers();
        let k = self.cfg.k_total();
        let m = self.cfg.n_targets;
        let lam = self.cfg.decay.factor(d_clock);

        if self.cfg.fit_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        // The clock each covariance has gone without an observation: every
        // row adds to it, and a row that observes the covariance's target
        // charges it and clears it, below (the module doc, task 211).
        for e in &mut self.elapsed {
            *e += d_clock;
        }
        // The clock since the moments last moved, this row's step in: what
        // the moments' floor below reads as the gap, the same number however
        // zero-weight rows split it (hard rule 9). A row that observes no
        // target moves the moments all the same, so it closes the gap
        // (review 2026-11-23, KAL-6).
        self.moments_gap += d_clock;
        let gap = self.moments_gap;

        // Which slots hold anything, and the map from the anchor to the
        // moments as they stand: formed after the last row that moved them,
        // or now after a load.
        if self.cfg.standardize {
            self.mark_live::<FACTOR>();
            if !self.map_fresh {
                self.form_map();
            }
        }

        // The clock has moved by `d_clock` since the last row: the state
        // is propagated before it predicts (`predict` does the same).
        self.transition::<FACTOR>(d_clock);

        // The row in the anchor's coordinates, read against the moments
        // BEFORE this row's update.
        let mut zs = std::mem::take(&mut self.zs);
        self.anchored_into(&self.zbuf, &self.live, &mut zs);
        self.zs = zs;

        // ---- predict (state before the update) ----
        // `z . b_j` per target, once, before any target's update: the
        // prediction and the innovation both read it, and so does the noise
        // `share_p` takes before its first residual, from every target's.
        for (zb, b) in self.zb.iter_mut().zip(&self.beta) {
            *zb = self.zs.iter().zip(b).map(|(z, b)| z * b).sum();
        }
        // The filter's own prediction, which its residual variance learns
        // from: on every row it predicts, whatever `min_weight` says, which
        // withholds the output alone (task 278, D10). A state written
        // before schema 64 learns from the output's, which the gate held
        // back (`KalmanCfg::min_weight_holds_learning`).
        let n_eff = self.stats.n_eff();
        let ready = n_eff >= self.cfg.min_weight;
        let pred = &mut out.pred;
        pred.clear();
        pred.resize(m, f64::NAN);
        if ready || !self.cfg.min_weight_holds_learning {
            for (j, p) in pred.iter_mut().enumerate() {
                if self.predicts_as::<FACTOR>(j) {
                    *p = a_number_or_none(self.zb[j]);
                }
            }
        }
        // The forecasts at the horizons, from the state as the prediction
        // reads it: after the row's transition, before its noise and update.
        // They ride after the predictions once the row is learned.
        let ahead = self.ahead_of_the_step(pred);

        // A null target, or a present one at weight zero -- an observation
        // of infinite variance, `σ²/0` -- is a prediction step and no
        // update, and time passes for both weights alike. The zero weight
        // skipped the decay, so `σ²`, which sets `R` and `Q`, forgot less
        // across it than across a null (review 2026-09-12, S9).
        let obs = |j: usize| y[j].filter(|_| weight > 0.0);

        // ---- the noise per target ----
        // `R`, and the `σ²` the process noise is derived from: `obs_var`,
        // else the residual variance, else -- before there is one -- the
        // row's own innovation squared (CC4). 0 is no noise to weigh the
        // row against: it corrects nothing and adds no derived `Q`. Under
        // `share_p`, the mean residual variance over the targets that have
        // one, as the row arrives, read once before any target's update
        // moves its own (review round 4, CC3), summed in ascending order so
        // the order of `targets` cannot move a bit of it (task 204), over
        // the targets with a variance (review round 5, A1); before any has
        // one, the mean squared innovation over the targets the row observes.
        let shared_s2 = if self.cfg.share_p {
            let mut buf = std::mem::take(&mut self.sumbuf);
            let s2 = shared_noise(&mut buf, &self.sig2);
            self.sumbuf = buf;
            s2
        } else {
            f64::NAN
        };
        let mut shared_first: Option<f64> = None;
        for j in 0..m {
            let sigma2 = match self.cfg.obs_var {
                Some(v) => v,
                None => {
                    let s2 = if self.cfg.share_p {
                        shared_s2
                    } else {
                        self.sig2[j]
                    };
                    if s2 > 0.0 {
                        s2
                    } else if self.cfg.share_p {
                        *shared_first.get_or_insert_with(|| self.shared_first_noise(y, weight))
                    } else {
                        obs(j).map_or(0.0, |yj| {
                            let e = yj - self.zb[j];
                            first_noise(e * e)
                        })
                    }
                }
            };
            self.noise[j] = sigma2;
        }

        // ---- the squared innovations a standardized prior is sized from ----
        // Under `share_p` the row's squares over the targets it observes,
        // summed in ascending order, so the order of `targets` moves no bit.
        // In units of the noise the prior reads no noise, and none is kept.
        if self.cfg.standardize && self.cfg.obs_var.is_none() && !self.cfg.noise_units {
            let square = |j: usize| {
                let e = obs(j)? - self.zb[j];
                Some(e * e).filter(|e2| *e2 > 0.0 && e2.is_finite())
            };
            if self.cfg.share_p {
                let mut buf = std::mem::take(&mut self.sumbuf);
                let sum = sum_ascending(&mut buf, (0..m).filter_map(square));
                let n = buf.len() as f64;
                self.sumbuf = buf;
                if n > 0.0 {
                    self.basis[0].add(sum, n, weight);
                }
            } else {
                for (j, basis) in self.basis.iter_mut().enumerate().take(m) {
                    if let Some(e2) = square(j) {
                        basis.add(e2, 1.0, weight);
                    }
                }
            }
        }

        // ---- process noise and priors, per covariance ----
        // Only on a row that observes the covariance's target (any target,
        // under `share_p`): `Q D²` for the whole clock since the last such
        // row, which a split of it into rows does not move (the module doc,
        // task 211). The noise is defined in the current coordinates, the
        // prior on a slot is set once its scale is usable, and an unsized
        // slot takes no noise (CC4).
        let n_p = self.p.len();
        for pi in 0..n_p {
            let informs = if self.cfg.share_p {
                (0..m).any(|j| obs(j).is_some())
            } else {
                obs(pi).is_some()
            };
            if !informs {
                continue;
            }
            let gap = self.elapsed[pi];
            self.elapsed[pi] = 0.0;
            let sigma2 = self.noise[if self.cfg.share_p { 0 } else { pi }];
            if self.cfg.standardize {
                self.add_process_noise::<FACTOR>(pi, sigma2, gap);
                let seeded = if self.units() {
                    // `p0` in units of the noise on each slope whose scale
                    // is usable; the intercept stays diffuse for the update.
                    self.size_slots_as::<FACTOR>(pi, 1.0, false, true)
                } else if let Some(r) = self.cfg.obs_var.or_else(|| self.basis[pi].noise()) {
                    self.size_slots_as::<FACTOR>(pi, r, true, true)
                } else {
                    false
                };
                if seeded {
                    self.refresh_zb_for_cov(pi);
                }
            } else if self.units() {
                self.add_process_noise::<FACTOR>(pi, sigma2, gap);
            } else if self.is_unsized_as::<FACTOR>(pi) {
                // The first row with a noise sizes the prior, to `p0` times
                // that noise, before its gain, and adds no process noise.
                self.size_whole(pi, self.cfg.p0 * sigma2);
            } else {
                self.add_process_noise::<FACTOR>(pi, sigma2, gap);
            }
        }

        // The smoother's slot for each observed covariance's last
        // observation takes this one's prior (`kalman/smooth.rs`).
        if self.smoother.is_some() {
            self.smooth_link(y, weight);
        }

        // ---- Kalman update per target ----
        // Under `share_p`, `P z` is read once, from `P` as the row finds it,
        // and `P` takes the row once, after every target's update (the
        // module doc, task 204).
        let mut shared_pz = false;
        let mut zpz_kept = 0.0;
        let mut shared_update = false;
        // In units of the noise the gain reads no noise, `s = z' P z + 1 /
        // w`, and a covariance whose intercept is unsized takes its first
        // row as a diffuse prior's exact step (the module doc). Whether it
        // is open is read before any target's update moves `P`.
        let units = self.units();
        let diffuse = units && self.cfg.fit_intercept;
        let open_shared = diffuse && self.cfg.share_p && !self.sized::<FACTOR>(0, 0);
        let mut extra_vals = innovation::init_extra(&mut out.extra, self.cfg.extra_width());
        let mut shared_s = f64::NAN;
        let mut shared_r = f64::NAN;
        for (j, &pred_j) in pred.iter().enumerate() {
            let pi = if self.cfg.share_p { 0 } else { j };
            let Some(yj) = obs(j) else {
                self.wj[j] *= lam;
                self.wsig[j] *= lam;
                continue;
            };
            let sigma2 = self.noise[j];
            // Unshared, target `j`'s covariance moves only below.
            let open =
                open_shared || (diffuse && !self.cfg.share_p && !self.sized::<FACTOR>(pi, 0));
            // pz = P z
            if !shared_pz {
                zpz_kept = self.read_pz::<FACTOR>(pi);
                shared_pz = self.cfg.share_p;
            }
            let zpz = zpz_kept;
            let r_inn = if units { 1.0 } else { sigma2 };
            let s_inn = zpz + r_inn / weight;
            let mut err = yj - self.zb[j];
            // A standardized regressor can be ~1e200 when a feature at the
            // input bound follows a run at a tiny scale, and then `z P z` or
            // `z . beta` overflows. The row is skipped rather than let an
            // `inf` gain or an `inf/inf` NaN into `beta` and `P`, which no
            // later row would repair (docs/IMPROVEMENTS.md C2). So is a row
            // with no noise yet to weigh it against: at `R = 0` the update
            // would take it as exact and collapse `P` along `z` (CC4).
            if r_inn > 0.0 && s_inn > 0.0 && s_inn.is_finite() && err.is_finite() {
                if let Some(ev) = extra_vals.as_deref_mut()
                    && ready
                    && !open
                    && !(diffuse && self.wj[j] == 0.0)
                    && !self.is_unsized_as::<FACTOR>(pi)
                {
                    self.record_innovation(j, zpz, s_inn, err, weight, shared_s2, ev);
                }
                if diffuse && (open || self.wj[j] == 0.0) {
                    // The diffuse intercept's gain is `e_0`: it takes the
                    // whole innovation, the level, and nothing else moves.
                    // A target first seen after its shared covariance took
                    // its first row is centred the same way.
                    self.beta[j][0] += err;
                    err = 0.0;
                }
                for i in 0..k {
                    self.gain[i] = self.pz[i] / s_inn;
                }
                if !open {
                    for (b, g) in self.beta[j].iter_mut().zip(&self.gain) {
                        *b += g * err;
                    }
                }
                if self.cfg.share_p {
                    shared_update = true;
                    shared_s = s_inn;
                    shared_r = r_inn / weight;
                } else {
                    self.take::<FACTOR>(pi, open, s_inn, r_inn / weight);
                }
            }
            // EW residual variance from the out-of-sample prediction. Its
            // weight ages on every row, this one included, and the row adds
            // its squared residual when it has a prediction to measure one
            // from: a row with no prediction -- the covariance not yet
            // sized, or, in a state written before schema 64, `min_weight`
            // unmet after a clock gap -- aged nothing, so `σ²` forgot less
            // across it than across a null (N6). The update is skipped when it would not be
            // finite: `sig2` feeds the process noise, and an `inf` there puts
            // `inf` on the diagonal of `P` and a NaN in every later gain.
            let aged = lam * self.wsig[j];
            self.wsig[j] = aged;
            if pred_j.is_finite() {
                let resid = yj - pred_j;
                let ws_new = aged + weight;
                let s2 = (aged * self.sig2[j] + weight * resid * resid) / ws_new;
                if s2.is_finite() {
                    self.sig2[j] = s2;
                    self.wsig[j] = ws_new;
                }
            }
            self.wj[j] = lam * self.wj[j] + weight;
        }
        // Every target that updated read the same `P z` and the same noise,
        // so the same gain: `P` takes the row once.
        if shared_update {
            self.take::<FACTOR>(0, open_shared, shared_s, shared_r);
        }
        if self.smoother.is_some() {
            self.smooth_observe(y, weight, ready);
        }

        // Standardization stats update last, so this row's z used the prior
        // stats; a row that moved them is followed by the drift check, and
        // past it the re-map (the module doc). A row of weight 0 moves no
        // moment, and the map stays the one formed after the last row that
        // did.
        // More than a half-life after the moments last moved, the
        // moments keep the row's own weight of history (the module doc,
        // *After a long gap*); the weight kept, `n_eff`, is the plain
        // recursion's (hard rule 8).
        if self.keeps_history(gap) {
            // The history's weight before the gap: what it holds now over
            // the decay since the last row moved it, zero-weight rows in it
            // or not (`inf` past the doubles, where nothing caps the floor).
            let had = lam * self.stats.n_eff() / self.cfg.decay.factor(gap);
            self.stats
                .update_keeping(&self.zbuf, lam, weight, weight, had);
        } else {
            self.stats.update(&self.zbuf, lam, weight);
        }
        if weight > 0.0 {
            self.moments_gap = 0.0;
        }
        if self.cfg.standardize && weight > 0.0 {
            self.follow_the_moments();
        }

        // A covariance whose diagonal is no longer a set of numbers of at
        // least 0 has lost its positive-definiteness, which no later row
        // repairs: sized again from the noise as it stands, as at the start
        // (the module doc; docs/PLAN.md task 214). After the row's
        // prediction, which reads `b` alone, so no prediction reads it.
        for pi in 0..self.p.len() {
            if self.lost_definiteness::<FACTOR>(pi) {
                self.size_again(pi);
            }
        }

        pred.extend_from_slice(&ahead);
        // `min_weight` withholds the output; the filter learned from it.
        if !ready {
            pred.fill(f64::NAN);
        }
        out.n_eff = n_eff;
    }
}

#[cfg(test)]
mod tests;
