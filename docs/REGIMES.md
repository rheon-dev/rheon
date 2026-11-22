<!-- kind: reference -->
# Regimes: what the detectors find, measured

This reference page is for anyone about to trust a regime or correlation
model: it gives what each finds on streams whose truth is known, and on
Monte-Carlo draws whose null distribution is published, where each is
right, and what it misses. A regime detector makes a claim about a stream:
that its correlation has changed, how long the current regime has lasted,
or which state a row belongs to. A claim like that can only be checked on
a stream whose answer is written down.
[`rp.sim.regimes`](how-to/test-a-detector-on-known-truth.md) generates one
from a seed and returns the truth beside the rows.

Five models make a claim about a stream: `deco`, `rcov`, `hmm`,
`corrchange` and `bocpd`. This page measures all five. It also measures two
ways out for [series that tick at their own
times](how-to/align-series-that-tick-at-their-own-times.md): refresh-time
sampling, and the lagged co-moments of `ew_cov(lags=...)`. Each model's
entry in [MODELS.md](MODELS.md) gives its update rule and a recipe that
runs it on a planted answer.

| section | the question | measured on |
|---|---|---|
| [0. The short version](#0-the-short-version) | what the experiments found, one paragraph each | |
| [1. Does `hmm` recover the stream that made it?](#1-does-hmm-recover-the-stream-that-made-it) | how many rows the filter puts in their true state, by how it was started, and how fast it follows a switch | `hmm` |
| [2. Is the monitor the size its paper says?](#2-is-the-monitor-the-size-its-paper-says) | how often the test rejects when nothing changed | `corrchange`, `kind="monitor"` |
| [3. And is it the power its paper says?](#3-and-is-it-the-power-its-paper-says) | how often it rejects a real break | `corrchange`, `kind="monitor"` |
| [4. Where the heavy tails go: `D-hat`](#4-where-the-heavy-tails-go-d-hat) | how well the statistic's denominator is estimated | `corrchange`, `kind="monitor"` |
| [5. Two changepoint detectors on the same break](#5-two-changepoint-detectors-on-the-same-break) | false alarms on a quiet stream, and the delay to a real break | `corrchange(kind="window")`, `bocpd` |
| [6. The Epps effect, and the two ways out](#6-the-epps-effect-and-the-two-ways-out) | what asynchrony does to a correlation, and what recovers it | `rp.stream.refresh_time`, `ew_cov(lags=...)` |
| [7. `deco`: an equicorrelation that moves](#7-deco-an-equicorrelation-that-moves) | where `rho` levels off, and how fast it follows a switch | `deco` |
| [8. `rcov`: three estimators against each block's truth](#8-rcov-three-estimators-against-each-blocks-truth) | each estimator's error under noise and asynchrony | `rcov` |
| [9. The sequential detector against its paper](#9-the-sequential-detector-against-its-paper) | how often a monitoring period ends in a flag when nothing changed | `corrchange`, `kind="sequential"` |
| [10. Running them](#10-running-them) | how to reproduce every number, and how long it takes | |

**Every number comes from `scripts/regime_experiments.py`, which
generates its streams from seeds** ([10. Running them](#10-running-them)).
Sections 1, 6 and 7 were run on 2026-09-27, and section 5 on 2026-10-26,
after the window test's critical value moved to the standard permutation
rule. Sections 2 to 4 and 9 were
run on 2026-09-28, after the monitor's long-run variance moved to its
paper's kernel weights, `1 − l/γ`. Section 8 was run on 2026-10-05, after the
pre-averaging bias's θ was read from the window the run uses and its first
pre-averaged term was formed as Christensen, Kinnebrock and Podolskij form
it. That moved two of its cells at the third decimal. Sections 2 to 4 and 9
were run again on 2026-11-22, after the monitor became the linear form of
its test and the sequential detector compared its correlations on Fisher's
scale; section 4's figures did not move, since the denominator is the same.

## 0. The short version

**Seeded from the rows, `hmm` splits zero-mean rows by direction.** Its
default seeding is k-means over the rows, and two zero-mean states differ
in nothing k-means can see. On streams that hold a single regime, the
filter puts 55 % of the rows in the true state, counting the 500 it reports
nothing on before seeding. Given the covariances to start from, it puts
98 % there. On streams that switch, it holds each new state within a
median of 12 rows, where seeded from the rows it follows no switch at all
([§1](#1-does-hmm-recover-the-stream-that-made-it)).

**Set each prior from the data's scale, because one set too wide fails the
same way in `hmm` and in `bocpd`.** At 1.0, on returns whose variance is
about 1.0, `hmm`'s `precision_prior` halves every correlation. The share of
rows in the true state then falls from 98 % to 58 %. Set too large,
`bocpd`'s `covariance_prior` makes every row unsurprising, so a real break is
never found ([its entry in
MODELS.md](MODELS.md#bocpd--how-long-has-this-regime-lasted)).
[§1](#1-does-hmm-recover-the-stream-that-made-it) measures the `hmm`
case.

**`corrchange`'s monitor is under its nominal size, and as powerful as its
paper.** Wied, Krämer and Dehling (WKD) tabulate its size and power for
"i.i.d. bivariate `t_5` innovations". The monitor here is the linear form of
their test, the CUSUM their proof reduces the statistic to. At 20,000 draws
a cell its size is 0.029 to 0.046 against the nominal 0.05, at or under
their Table 1 in every cell, and its size-adjusted power is within 0.023 of
their Table 2 on the `t_5` design
([§2](#2-is-the-monitor-the-size-its-paper-says),
[§3](#3-and-is-it-the-power-its-paper-says)).

**`D-hat`, the statistic's denominator, is exact on Gaussian pairs and
biased low on a tail-dependent `t_5`.** On Gaussian pairs it averages 0.750
against an asymptotic 0.750 at `T = 2000`. On the `t_5` it reads 18 % low
at `T = 500` and 9 % low at `T = 2000`, and its scatter grows with `T`
instead of shrinking. The monitor's size and power stay close to WKD's
even so ([§4](#4-where-the-heavy-tails-go-d-hat)).

**`corrchange`'s sequential detector holds its size at every boundary.**
Wied and Galeano tabulate the rate at which a monitoring period ends in a
flag when nothing changes, on GARCH pairs, and at `boundary_gamma` 0.45
theirs reads 0.125 to 0.174 against a nominal 0.05. The detector here
compares the correlations on Fisher's scale, and over about 1000 cycles a
cell it reads 0.040 to 0.074 at every `boundary_gamma`, 0.050 to 0.068 at
0.45 ([§9](#9-the-sequential-detector-against-its-paper)).

**The two changepoint detectors are at opposite ends of one trade-off.** On
the same streams, `corrchange(kind="window")` finds a correlation break in a
median of 29 rows and flags 45 rows per 1000 quiet ones. `bocpd` takes 121
rows and gives 0.53 alarms per 1000. Neither is better: they answer
different questions ([§5](#5-two-changepoint-detectors-on-the-same-break)).

**Asynchrony leaves almost nothing of a correlation at the finest interval,
and both ways out recover most of it.** A true correlation of 0.8 reads as
0.095 from one-row returns. Refresh-time sampling recovers 0.54 from 19157
returns, 16 % of the 119991 at the finest interval. The lag inversion of
§6, on `ew_cov(lags=...)`, recovers 0.77 while reading every return.
Averaged over five seeds it levels off at 0.775 for lags from 256 to 512
rows ([§6](#6-the-epps-effect-and-the-two-ways-out)).

**`deco`'s estimate levels off near two-thirds of the true equicorrelation, and
moves at its half-life.** The row's estimate is biased low, as its paper
warns, and the level is an exponentially weighted mean of it
([§7](#7-deco-an-equicorrelation-that-moves)).

**Each of `rcov`'s estimators wins somewhere.** `plain` is the most
accurate on clean returns, the kernel under noise, and pre-averaging when
the series are observed on different rows
([§8](#8-rcov-three-estimators-against-each-blocks-truth)).

## 1. Does `hmm` recover the stream that made it?

`hmm` gives each row a probability for each of `k` hidden states. Here
the filter is started in four ways, and each is scored by how many rows it
puts in the state that generated them.

**No stream in this first experiment switched regime.**
`rp.sim.regimes` draws one state per block, and with ten blocks at a 1 %
chance per block of leaving state 0, every block of all eight seeds stayed
there. So the first results table gives the share of rows the filter puts
in that one true state. [The switch experiment](#through-a-switch) makes
the streams switch.

| condition | value |
|---|---|
| streams | 8 seeds of `rp.sim.regimes`: two series, 10 blocks of 400 rows, 4000 rows each, differenced into 3999 returns |
| states drawn | correlation 0.1 and 0.8, under the sticky transition matrix `[[0.99, 0.01], [0.02, 0.98]]` |
| states realized | state 0, at correlation 0.1, in every block of every seed |
| the filter | `k = 2`, `half_life = 4000`, and `precision_prior = 0.01` except in the last row |
| seeded from the rows | `warm_rows = 500` and `transition_prior = 200` |
| started from the truth | the true transition matrix, zero means, and covariances built from the true correlations and each seed's sample variances |
| the score | the share of rows in state 0, read as `filtered_state1` at most 0.5, with a row that has no `filtered_state1` counted there. The script takes the better of the two ways to match states to the truth, which here is this share |

| how it was started | mean share in the true state | worst seed |
|---|---|---|
| seeded from the rows (`warm_rows`), learning | 0.553 | 0.505 |
| true covariances, filter only | 0.979 | 0.966 |
| true covariances, and go on learning | 0.976 | 0.961 |
| true covariances, filter only, `precision_prior` = 1.0 | 0.580 | 0.521 |

**Seeded from the rows, the filter splits zero-mean rows by direction.**
The default seeding is k-means over the rows, and these returns have a
mean of zero, so k-means finds two directions. The learned state means
reach 0.64 in absolute value, where the truth is zero. That is the largest
absolute mean per seed, averaged over the eight. The score counts the 500
rows before seeding, which report nothing, as the true state. Over the 3499
rows after seeding the share is about half: (0.553 × 3999 − 500) / 3499 ≈
0.49. So a regime that lives in the covariance needs the covariances to
start from, or a feature in which the regime is a shift in location.

**A ridge as large as the data's variance halves every correlation.** In
the last row the filter does not learn, so `precision_prior` adds a ridge
of 1.0 to each given covariance for the whole stream. The returns' variance
is about 1.0, so each state's correlation is halved, and the correlation is
the whole of the signal here. The share in the true state falls from 0.979
to 0.580. So keep `precision_prior` well below the data's variance, as
[`hmm`'s entry in MODELS.md](MODELS.md#hmm--which-regime-are-we-in) says.
`bocpd`'s `covariance_prior` fails the same way when set too large ([its
entry](MODELS.md#bocpd--how-long-has-this-regime-lasted)).

**Learning from the true covariances keeps the filter's share, because
each state forgets toward the covariance it was given.** A learning model's
states forget toward a prior worth `prior_rows` rows, by default the number
of features, 2 here. With the states given, each state's prior is its own
given covariance, rescaled as the rows' variances move
([`hmm`'s entry in MODELS.md](MODELS.md#hmm--which-regime-are-we-in)). The
state that wins no rows here, the one at correlation 0.8, goes on reading
0.8, so the rows that fit neither state well do not teach it the present
regime. The share learning keeps is 0.976 against the filter's 0.979, and
0.961 in the worst seed against 0.966.

The learned transition matrix is not read back through the public API, so
what is scored is the state path, which is what a caller reads. The
matrix's decayed counts are in the state that `ModelBank.write_json()`
exports, but no method returns the matrix.

### Through a switch

**Started from the true covariances, the filter follows each switch within
a few dozen rows.** `durations=[3, 3]` makes each state last exactly three
blocks, so each stream alternates between correlation 0.1 and 0.8 and
switches three times. A switch counts as followed once the filter holds the
new state for 20 rows running. That excludes a filter that flips at random,
which agrees with either state within a row or two.

| condition | value |
|---|---|
| streams | 8 seeds of `rp.sim.regimes` with `durations=[3, 3]`: two series, 12 blocks of 400 rows, 4799 returns each, switching at rows 1200, 2400 and 3600, 24 switches in all |
| the filter | as above, but the true start's transition is the per-row hazard `1/1200` |
| the score | the share of rows in the true state, a row with no `filtered_state1` counted as a miss; and the rows after each switch until the filter holds the new state for 20 rows |

| how it was started | mean share in the true state | worst seed | median rows to hold the new state | longest | switches never held |
|---|---|---|---|---|---|
| seeded from the rows (`warm_rows`), learning | 0.456 | 0.450 | - | - | 24 of 24 |
| true covariances, filter only | 0.992 | 0.989 | 12 | 32 | 0 of 24 |
| true covariances, and go on learning | 0.987 | 0.979 | 12 | 35 | 0 of 24 |

**Seeded from the rows, it follows no switch.** It never holds either
state for 20 rows running, because it splits the rows by direction, as in
the table above. Its share is below a half because here the 500 rows before
seeding count as misses.

**Learning follows a switch about as fast.** With learning on, the median
switch is held within 12 rows and the longest took 35, and the worst seed's
share is 0.979. Without learning, no switch took more than 32 rows, and no
seed's share fell below 0.989. A state whose regime is away for 1,200 rows
reads the covariance it was given when the regime comes back.

## 2. Is the monitor the size its paper says?

`kind="monitor"` is the linear form of Wied, Krämer and Dehling's
closed-sample constancy test: were the correlations constant over a span of
rows? It is the CUSUM of the delta-method series their proof reduces their
statistic to, against the same limit, read on a grid of `T` points. A
published test comes with a published null, and its size is the rate at
which it rejects when the correlation never changes. WKD's Table 1 gives that rate for
"i.i.d. bivariate `t_5` innovations". The phrase does not say whether the
two coordinates share a scale, and the two readings differ in their tails,
so both are measured, with Gaussian pairs beside them:

| draw | how it is made | its tails |
|---|---|---|
| t5 shared scale | a Gaussian pair divided by one chi-square scale per row, shared by both coordinates, which leaves the correlation at `rho` | tail-dependent: the extremes arrive in both coordinates at once |
| t5 independent | the same pair, each coordinate divided by its own scale | the same marginals, with no tail dependence |
| Gaussian | the pair itself | |

| T    | rho  | WKD Table 1 | t5 shared scale | t5 independent | Gaussian |
|------|------|-------------|-----------------|----------------|----------|
| 500  | -0.5 | 0.040       | 0.030           | 0.035          | 0.044    |
| 500  | 0.0  | 0.035       | 0.029           | 0.037          | 0.041    |
| 500  | 0.5  | 0.041       | 0.029           | 0.035          | 0.042    |
| 1000 | -0.5 | 0.038       | 0.031           | 0.040          | 0.042    |
| 1000 | 0.0  | 0.034       | 0.032           | 0.040          | 0.043    |
| 1000 | 0.5  | 0.039       | 0.033           | 0.036          | 0.046    |

The table holds 20,000 replications per cell, at a nominal level of 0.05,
where a rate from 20,000 draws has a standard error of 0.0015. Each draw is
one span of `T` rows, tested once with `alpha_adjust="none"`.

**Under the shared-scale reading, the size is 0.029 to 0.033, at or under
WKD's Table 1 in every cell.** The linear form is the more conservative of
the two on heavy tails. Its sums and its denominator are made of the same
rows, so a few large rows lift the denominator as much as the sums they
enter. And it forms no correlation of a span's first few rows, where WKD's
statistic gained rejections at every distribution.
Neither reading is liberal: no `t_5` cell is above 0.040.

**On Gaussian pairs the size is 0.041 to 0.046**, a little under the
nominal 0.05. `tests/test_corrchange.py` holds the test to that level on
Gaussian pairs at `T = 500`, for `rho` of 0 and 0.5, and again at 0.5 with
one column multiplied by 100. Its extended tier holds the size at 8 to 1000
rows, at correlations up to 0.9, on `t_5`, GARCH and scales that double.
There WKD's own statistic read up to 0.26, at a correlation of 0.9 and 100
rows.

## 3. And is it the power its paper says?

Power is the rate at which the test rejects when the correlation does
change. WKD's Table 2 gives it for a break from 0.5 to 0.7 at the middle of
the span. The same three draws are measured, each against two critical
values:

| column | the draw is rejected when its statistic exceeds |
|---|---|
| power | the model's critical value, the asymptotic 1.3581 less `0.5826/√T` |
| size-adjusted | the empirical 95 % quantile of 1000 draws of the same distribution and length, with the correlation held at 0.5 |

| T    | innovations | power | size-adjusted | empirical 95% | WKD Table 2 |
|------|-------------|-------|---------------|---------------|-------------|
| 500  | t5          | 0.533 | 0.589         | 1.267         | 0.587       |
| 500  | t5_indep    | 0.615 | 0.653         | 1.296         |             |
| 500  | normal      | 0.855 | 0.863         | 1.316         |             |
| 1000 | t5          | 0.830 | 0.853         | 1.298         | 0.830       |
| 1000 | t5_indep    | 0.865 | 0.871         | 1.316         |             |
| 1000 | normal      | 0.997 | 0.997         | 1.348         |             |

The table holds 1000 replications per row, at a nominal level of 0.05.

**On the shared-scale `t_5` the size-adjusted power is WKD's:** 0.589
against their 0.587 at `T = 500`, and 0.853 against 0.830 at `T = 1000`.
Against the model's own critical value it reads 0.533 and 0.830, lower at
500 rows by what its under-nominal size costs there. The independent draw
reaches 0.615 and 0.865, or 0.653 and 0.871 size-adjusted.

**On Gaussian pairs the power is well above WKD's figure:** 0.855 at
`T = 500`, or 0.863 size-adjusted, and 0.997 at `T = 1000`.
`tests/test_corrchange.py` holds the Gaussian power at `T = 500` above a
floor set from WKD's 0.587, less a tolerance.

**Read the size-adjusted column across distributions.** It takes its
critical value from null draws of the same distribution and length, so a
size that differs between distributions does not tilt it. Every empirical
quantile, from 1.267 to 1.348, is below the model's critical value (1.332
at 500 rows, 1.340 at 1000). So the size-adjusted power is the higher of
the two in every row but the last, where both are 0.997.

## 4. Where the heavy tails go: `D-hat`

The monitor's statistic, `Q`, is
`max_j |sum_{t<=j} xi_t| / (sqrt(T) D-hat)`, the CUSUM of the
delta-method series `xi_t = x_t y_t - (rho/2)(x_t^2 + y_t^2)` on the
standardized span. `D-hat`, the denominator, is the delta-method long-run
standard deviation of `sqrt(T)*rho-hat`, the Bartlett long-run deviation of
`xi`. For an
elliptical distribution, the value it estimates is known in closed form:
`(1 - rho^2) sqrt(1 + kappa)`, with `kappa = 2/(nu - 4)`. At `rho` = 0.5
that is 0.750 for a Gaussian pair, and 1.299 for a `t_5`, whose `kappa`
is 2.

The experiment recovers `D-hat` from each draw as the numerator, computed
longhand, divided by the reported `Q`, so it is the estimate the model
actually used.

| innovations | T    | mean  | median | sd    | asymptotic |
|-------------|------|-------|--------|-------|------------|
| normal      | 500  | 0.741 | 0.743  | 0.058 | 0.750      |
| normal      | 2000 | 0.750 | 0.750  | 0.031 | 0.750      |
| t5          | 500  | 1.071 | 1.016  | 0.255 | 1.299      |
| t5          | 2000 | 1.185 | 1.099  | 0.353 | 1.299      |
| t5_indep    | 500  | 0.812 | 0.795  | 0.111 | -          |
| t5_indep    | 2000 | 0.828 | 0.816  | 0.072 | -          |

The table holds 200 replications per row, at `rho` = 0.5. The independent
`t_5` has no closed form here, so its last column reads `-`.

**On Gaussian pairs the estimator is right and tight.** It averages 0.741
at `T = 500` and 0.750 at `T = 2000`, against 0.750. Its standard deviation
falls from 0.058 to about 0.03 (0.031) as `T` grows fourfold. That is what
the delta method promises on data with light tails: right, consistent and
quick.

**On the shared-scale `t_5` it is biased low, and its scatter does not
shrink.** Its mean is 1.071 at `T = 500` and 1.185 at `T = 2000`, against
1.299: 18 % and 9 % low. Its median is lower still, 22 % and 15 % low, and
its standard deviation grows from 0.255 to 0.353. The estimator's inputs
are fourth moments, and a `t_5` has a fourth moment only just, at
`nu > 4`. Those inputs' own variance is infinite, so there is no rate at
which the estimate steadies. The independent `t_5` is steadier: its
standard deviation falls from 0.111 to 0.072.

The size in §2 and the power in §3 are close to WKD's tables even so.

## 5. Two changepoint detectors on the same break

Both detectors read the same 20 pairs of streams from `rp.sim.regimes`,
each stream three series of 2000 rows. In each pair, one stream keeps a
correlation of 0.2 throughout, and the other breaks from 0.2 to 0.7 at row
1000. A false alarm is an alarm on the quiet stream, and the delay is the
number of rows from row 1000 to the first alarm at or after it.

| detector | settings | what counts as its alarm |
|---|---|---|
| `corrchange(kind="window")` | `kind="window"`, `window_rows=100`, `n_perm=100`, `permute_every_rows=100`, `alpha=0.05`, `seed=0` | `flag`: the change between two adjacent 100-row windows is above a permutation quantile |
| `bocpd` | `covariance_type="full"`, `hazard_rows=500`, `degrees_of_freedom_prior=5`, `covariance_prior=[0.1]`, `mean_prior` left out (the first 5 rows' mean), `prune_below=1e-8`, `max_run=1200` | its own alarm, `p_change`, cannot see this break, so here: the row that `t - run_mode` names as the current run's start jumps forward by more than 50 |

**`bocpd` is measured with `covariance_type="full"`, not its default
`"diag"`.** The default models each feature on its own, and only the
full covariance can see a break in the correlation alone ([its entry in
MODELS.md](MODELS.md#bocpd--how-long-has-this-regime-lasted)).

| detector           | false alarms / 1000 quiet rows | found the break | median delay | worst delay |
|--------------------|--------------------------------|-----------------|--------------|-------------|
| corrchange, `kind="window"` | 45.35                          | 20/20           | 29           | 56          |
| bocpd              | 0.53                           | 20/20           | 121          | 351         |

**The two detectors answer different questions, which is why both are
here.** `corrchange(kind="window")` is a test of one thing: whether the
correlation matrix has moved between two adjacent windows. It is looking
straight at this break, so it flags it fast, in a median of 29 rows and 56
at worst. It also flags 45.35 rows per 1000 quiet ones. Two windows that
slide by one row are almost the same windows, so its flags come in runs.
That rate is nearer a fraction of rows above the quantile than a count of
events.

`bocpd` is a full joint model of the rows, and a correlation change is the
one break its `p_change` cannot see: no single row is surprising, only the
sequence is. Its run-length posterior has to accumulate that evidence,
which takes a median of 121 rows and 351 at worst. In return it dates the
regime, saying which row the current one began on. And it almost never
says so when nothing has happened: 0.53 alarms per 1000 quiet rows.

**None of those alarms comes from the cap on a run's length.** The runs
that reach `max_run - 1 = 1199` rows fold into the one of them carrying the
most mass, scaled back to 1199 rows' weight: on a quiet stream the oldest,
and after a break the post-break run, once it reaches the cap. So the run
in progress keeps its place, and the start it reports moves a row at a
time. The capped run is then an exponentially weighted run, scaled by
`1199 / 1200` each row, with a half-life of `ln 2 / ln(1200 / 1199)`, about
830 rows.

**Folding into any other run at the cap gives alarms of its own, which is
why the rule follows the mass.** Folding into the youngest run at the cap
gives every quiet stream an alarm a few rows past row 1200, where the
run's reported start jumps to that younger run's. That takes the rate to
1.00 per 1000. The same alarm, at rows 1203 to 1207, is the first one
after the break on three broken streams, and makes the worst delay read
207. It is the cap firing on schedule, not the break being found: within
200 rows of the break both rules find it on 17 of 20 streams. Folding into
the oldest run gives every figure in this table too, since on these
streams the oldest run is the one carrying the mass. Where nothing is
pruned, it is not. After a 3-sigma mean shift, the post-break run reaches
the cap, and folding into the oldest run puts its mass onto the row-0
run's statistics, the old regime's mean at the cap's weight. The reported
start then jumps a second time about `max_run` rows after the break, on 17
of 20 streams at `max_run=200`. Neither the uncapped posterior on the same
rows nor the rule that follows the mass does so.

## 6. The Epps effect, and the two ways out

The Epps effect is the fall of a measured correlation toward zero as its
returns are taken over shorter intervals, when the series tick at their
own times. [Series that tick at their own
times](explanation/asynchronous-series.md) explains why it happens and
how much of it each way out recovers. The stream here has two series with a true
correlation of 0.8, over 120,000 rows, each observed at an expected rate
of 0.25 per row (`async_rates`). Previous-tick sampling carries each
series' last value forward and takes returns every so many rows. At the
finest interval it reads 0.095:

| sampling interval | returns | correlation |
|-------------------|---------|-------------|
| 1                 | 119991  | 0.095       |
| 2                 | 59995   | 0.172       |
| 5                 | 23998   | 0.346       |
| 20                | 5999    | 0.634       |
| 100               | 1199    | 0.752       |
| 500               | 239     | 0.745       |

The truth is 0.8. Two ways out recover most of it, and they differ in what
they read:

| way out | what it reads | correlation |
|---|---|---|
| refresh-time sampling, `rp.stream.refresh_time` | a grid point at the first row by which both series have ticked since the last point: 19157 returns, 16 % of the 119991 at the finest interval. On average a grid point kept 81 % of the ticks since the one before | 0.543 |
| lag inversion, `ew_cov(lags=...)` read through `rheon.corr.epps_invert` | every one of the 119991 fine returns, with no grid: the correlation at a scale of `L` rows, from the lagged co-moments at one row | 0.772 at `L` = 192 |

The lag inversion is read at 15 values of `L`, on this stream and on four
more from the next seeds, so its noise is printed beside it:

| L    | seed 7 | mean of 5 seeds | sd across seeds |
|------|--------|-----------------|-----------------|
| 1    | 0.095  | 0.101           | 0.004           |
| 2    | 0.170  | 0.178           | 0.005           |
| 4    | 0.294  | 0.301           | 0.005           |
| 8    | 0.456  | 0.457           | 0.003           |
| 16   | 0.608  | 0.602           | 0.004           |
| 32   | 0.699  | 0.697           | 0.006           |
| 64   | 0.742  | 0.744           | 0.005           |
| 96   | 0.757  | 0.758           | 0.003           |
| 128  | 0.765  | 0.767           | 0.001           |
| 192  | 0.772  | 0.774           | 0.004           |
| 256  | 0.770  | 0.775           | 0.008           |
| 384  | 0.766  | 0.775           | 0.015           |
| 512  | 0.765  | 0.774           | 0.019           |
| 768  | 0.760  | 0.771           | 0.026           |
| 1024 | 0.756  | 0.769           | 0.029           |

**The sweep passes the best point of `L`'s trade-off** ([The two ways
out](explanation/asynchronous-series.md#the-two-ways-out)). The mean over five
seeds levels off at 0.775 from `L` = 256 to 512 and falls after, while its
spread across seeds grows from 0.001 at 128 to 0.029 at 1024. Seed 7 peaks
at a smaller `L`, 192. Anywhere from 128 to 512 reads within 0.01 of the best,
0.775, which is still short of the truth of 0.8.

The lag inversion is here because its lagged co-moments are a state a bank
already keeps, as `ew_cov(lags=...)`. With `lags` 1 to 5 at 20 columns,
`ew_cov` ran 1.15M rows a second against 2.12M without them, measured on
2026-09-06: five more 20 × 20 outer products a row, and the ring of rows
they need. [Rows a second for each
model](PERFORMANCE.md#rows-a-second-for-each-model) has the current
figure.

## 7. `deco`: an equicorrelation that moves

`deco` reduces a correlation matrix to one number, the average of its
off-diagonal, and follows it with an exponentially weighted mean of each row's own estimate,
`u`. The stream here has five series whose equicorrelation alternates
between 0.2 and 0.6.

| condition | value |
|---|---|
| streams | 8 seeds of `rp.sim.regimes`: m = 5, `states=[0.2, 0.6]`, `durations=[3, 3]`, 12 blocks of 400 rows, 4799 returns each, 24 switches in all |
| the model | `rheon.spec.deco` with the default `dynamics="ew"`, at half-lives of 25, 100 and 400 rows |
| level | `rho` averaged over the last 200 rows of each stretch of one state |
| rows to the midpoint | the rows after a switch until `rho` passes the midpoint of the two levels |

| half-life | `rho`'s level, true 0.2 | `rho`'s level, true 0.6 | median rows to the midpoint | range | switches never crossed |
|---|---|---|---|---|---|
| 25 | 0.129 | 0.377 | 26 | 8–58 | 0 of 24 |
| 100 | 0.130 | 0.381 | 100 | 66–172 | 0 of 24 |
| 400 | 0.146 | 0.348 | 330 | 246–520 | 0 of 24 |

**`rho` levels off near two-thirds of the truth, because `u` does.** The
row's estimate `u` averages 0.130 where the truth is 0.2, and 0.379 where
it is 0.6. The spec's docstring warns of this, after Engle and Kelly: `u`
is a ratio, and its mean is not the ratio of the means. `rho` is an exponentially weighted mean
of `u`, so it carries the same bias. Read `rho` as a signal that moves with
the correlation, not as the correlation.

**It moves at its half-life.** An exponentially weighted mean reaches the midpoint of a step in
one half-life, and `rho` does: a median of 26 rows at a half-life of 25, and
100 at a half-life of 100. At 400 the median is 330. A state lasts 1200
rows, three of those half-lives, so `rho` has not reached its level when the next
switch comes. Its two levels, 0.146 and 0.348, are closer together than at
the shorter half-lives, and it starts nearer their midpoint.

## 8. `rcov`: three estimators against each block's truth

`rcov` estimates each block's covariance from its returns, at the block's
close. Its three estimators differ in what they survive: noise on the
observed prices, and series that are not observed on every row. The stream
here has 20 blocks of 2000 rows, each block's correlation drawn from 0.3
and 0.8, under four conditions:

| condition | what changes |
|---|---|
| clean | nothing: the returns are the latent ones |
| noise | noise with a standard deviation of 0.5 on each observed level (`noise=0.5`), which makes each return an MA(1) with a negative first autocorrelation, −0.17 here |
| noise, half observed, previous tick | the noise, and each series observed on half the rows (`async_rates=[0.5, 0.5]`), its last value carried forward within the block |
| noise, half observed, refresh time | the same rows through `rp.stream.refresh_time`, block by block |

Each cell is the bias of the block's correlation, its standard error, and
the root-mean-square error. There are 190 blocks: 10 seeds of 20 blocks,
less the last block of each stream, which never closes. `block_rows` sizes
the kernel's ring and pre-averaging's window. It is each condition's own
median rows a block: about 2000, and 602 under refresh time, which keeps
only the rows where both series have moved.

| condition | `plain` | `kernel` | `preavg` | `preavg`, `psd=False` |
|---|---|---|---|---|
| clean | −0.001 ± 0.001 / 0.016 | −0.004 ± 0.003 / 0.042 | −0.013 ± 0.008 / 0.113 | −0.006 ± 0.005 / 0.074 |
| noise | −0.177 ± 0.006 / 0.197 | −0.011 ± 0.003 / 0.045 | −0.013 ± 0.008 / 0.113 | −0.006 ± 0.005 / 0.074 |
| noise, half observed, previous tick | −0.420 ± 0.014 / 0.465 | −0.094 ± 0.004 / 0.110 | −0.017 ± 0.008 / 0.114 | −0.021 ± 0.005 / 0.078 |
| noise, half observed, refresh time | −0.219 ± 0.008 / 0.246 | −0.017 ± 0.005 / 0.075 | −0.020 ± 0.011 / 0.150 | −0.010 ± 0.008 / 0.106 |

**On clean returns, `plain` is the estimator to use.** Its error is 0.016,
against 0.042 for the kernel and 0.074 at best for pre-averaging. The
other two trade variance for robustness that clean data does not need.

**Noise takes about a third of `plain`'s correlation, and almost nothing of the
kernel's.** Noise adds variance to each series and nothing to their
covariance, so `plain` reads 0.177 low on average. The kernel's error moves
from 0.042 to 0.045. Pre-averaging does not move at all at this precision:
each block moves by about 0.003, and across 190 blocks those moves cancel
to the third decimal.

**Asynchrony is the harder case.** Carried forward, a series that was not
observed returns zero, and `plain` loses 0.420. The kernel, which sums its
lags, keeps most of it, at a bias of −0.094. Pre-averaging is the least
biased, at −0.017 and −0.021, because its window, tens of rows long, spans
the rows a series missed. Refresh time takes the kernel's bias to −0.017, on
602 rows a block, and pre-averaging's error rises there, to 0.150 and 0.106,
with fewer rows to average over.

**`psd=False` is the better pre-averaging form here.** It has the lower
error in every condition, 0.074 against 0.113 on clean data. `psd=True` is
the default because its estimate is always positive semi-definite: it
clips a negative eigenvalue where one appears, and the balanced form,
`psd=False`, does not promise that. For a correlation read on its own, the balanced form was
the more accurate on these streams.

## 9. The sequential detector against its paper

`kind="sequential"` is Wied and Galeano's (2013) monitoring procedure. It
takes `window_rows` rows of history, then tests each row of a monitoring
period of `monitor_rows` rows against it as the row arrives, until a flag
or the period's end. Their `T` is `monitor_rows / window_rows`, and
`boundary_gamma`, their `γ`, lowers the boundary early in the period. The
detector here compares the monitored rows' correlation with the history's
on Fisher's scale, each at its own finite-sample variance, where theirs
compared the correlations themselves from the second monitored row. Its
size is the share of monitoring periods that end in a flag when the
correlation never changes. Their Table 2 gives it for two independent
GARCH(1,1) series mixed to a correlation of 0.5, 1000 series a cell. Here
each cell is one stream of about 1000 back-to-back cycles on the same
design, and on Gaussian pairs beside it:

| gamma | T   | m   | W&G Table 2 | GARCH | Gaussian |
|-------|-----|-----|-------------|-------|----------|
| 0.0   | 0.5 | 250 | 0.059       | 0.056 | 0.054    |
| 0.0   | 0.5 | 500 | 0.058       | 0.053 | 0.053    |
| 0.0   | 1.0 | 250 | 0.077       | 0.064 | 0.076    |
| 0.0   | 1.0 | 500 | 0.069       | 0.061 | 0.063    |
| 0.0   | 2.0 | 250 | 0.066       | 0.060 | 0.058    |
| 0.0   | 2.0 | 500 | 0.054       | 0.055 | 0.054    |
| 0.0   | 4.0 | 250 | 0.063       | 0.040 | 0.051    |
| 0.0   | 4.0 | 500 | 0.071       | 0.062 | 0.048    |
| 0.25  | 0.5 | 250 | 0.075       | 0.052 | 0.058    |
| 0.25  | 0.5 | 500 | 0.079       | 0.054 | 0.048    |
| 0.25  | 1.0 | 250 | 0.075       | 0.074 | 0.064    |
| 0.25  | 1.0 | 500 | 0.064       | 0.060 | 0.061    |
| 0.25  | 2.0 | 250 | 0.087       | 0.068 | 0.062    |
| 0.25  | 2.0 | 500 | 0.063       | 0.064 | 0.063    |
| 0.25  | 4.0 | 250 | 0.073       | 0.061 | 0.049    |
| 0.25  | 4.0 | 500 | 0.077       | 0.065 | 0.054    |
| 0.45  | 0.5 | 250 | 0.169       | 0.051 | 0.054    |
| 0.45  | 0.5 | 500 | 0.125       | 0.053 | 0.050    |
| 0.45  | 1.0 | 250 | 0.174       | 0.060 | 0.062    |
| 0.45  | 1.0 | 500 | 0.136       | 0.050 | 0.050    |
| 0.45  | 2.0 | 250 | 0.164       | 0.057 | 0.053    |
| 0.45  | 2.0 | 500 | 0.138       | 0.061 | 0.054    |
| 0.45  | 4.0 | 250 | 0.161       | 0.058 | 0.054    |
| 0.45  | 4.0 | 500 | 0.128       | 0.068 | 0.038    |

The nominal level is 0.05 in every row. A rate from 1000 draws has a
standard error of about 0.008 at 0.07 and 0.011 at 0.15, and the
difference between two such rates about √2 times that.

**At `boundary_gamma` of 0 and 0.25 the detector is at or under their
table in all but two cells,** and those within a standard error and a half
of it. On their design it reads 0.040 to 0.074 against their 0.054 to
0.087, and on Gaussian pairs 0.048 to 0.076.

**At 0.45 it holds the nominal level where theirs did not.** Their table
reads 0.125 to 0.174, the boundary low at the start of the period and a
correlation of the first two monitored rows, `±1` whatever the data, set
against it. On Fisher's scale those rows carry no information and report
nothing, and the detector reads 0.050 to 0.068 on their design and 0.038 to
0.062 on Gaussian pairs. The low start still catches an early change
sooner, which `tests/test_corrchange.py` holds the detector to, ten rows
in.

**The critical values are within 0.03 of the paper's.** The paper
simulates them on a grid, which reads the supremum low. This library solves
the law behind them (`crates/rheon-core/src/boundary.rs`), so its values
are higher in 11 of the 12 cells of their Table 1, and within 0.03 of it in
all. They are read at the period's length in the detector's own time,
`(monitor_rows − 3)/(window_rows − 3)`, which is their `T` where the two
counts are equal.

`tests/test_corrchange.py` holds the Gaussian size at `m = 250`, `T = 1`
between 2.5 % and 7.5 %, and the critical values to the paper's Table 1.

## 10. Running them

Each experiment is one command of the script, named for what it measures,
and `all` runs every one:

```sh
uv run python scripts/regime_experiments.py all              # every experiment, in the order of the experiments table
uv run python scripts/regime_experiments.py size power dhat  # any of them by name, in the order given
```

| experiment | section | wall time |
|---|---|---|
| `recovery` | [§1](#1-does-hmm-recover-the-stream-that-made-it) | 0.2 s |
| `switch` | [§1](#through-a-switch) | 0.2 s |
| `size` | [§2](#2-is-the-monitor-the-size-its-paper-says) | 101.6 s |
| `power` | [§3](#3-and-is-it-the-power-its-paper-says) | 3.6 s |
| `dhat` | [§4](#4-where-the-heavy-tails-go-d-hat) | 0.5 s |
| `delay` | [§5](#5-two-changepoint-detectors-on-the-same-break) | 1.3 s |
| `epps` | [§6](#6-the-epps-effect-and-the-two-ways-out) | 3.9 s |
| `deco` | [§7](#7-deco-an-equicorrelation-that-moves) | 0.1 s |
| `rcov` | [§8](#8-rcov-three-estimators-against-each-blocks-truth) | 3.6 s |
| `sequential` | [§9](#9-the-sequential-detector-against-its-paper) | 27.2 s |

`all` runs them in that order in about 2.5 minutes of wall time, most of it
the size study's 20,000 draws a cell. That was measured on 2026-09-27, and
on 2026-11-22 for `size`, `power`, `dhat` and `sequential` (Python 3.14.6,
Polars 2.1.0, the development wheel, at a load of 5), on an Apple M4 Pro
(14 cores, 48 GB) with Python 3.12.13, Polars 1.44.2 and a release build. The times of `delay` and `rcov` come from those runs too,
before their sections were run again.

**Every stream is generated from a seed, so two runs of one build give the
same numbers.** The seeds are derived with `zlib.crc32`, because Python
randomizes a string's `hash()` in each process. Nothing downloads and
nothing is cached. A different build can move the numbers, and [the page's
opening](#regimes-what-the-detectors-find-measured) gives the date of each
section's run.

**A test holds sections 1 and 5 to 9 to what their experiments print.**
`tests/test_regimes_doc.py` runs those experiments, in the extended tier,
and checks every row of the sections' tables and every three-decimal figure
in their prose against the output. It runs on macOS alone, where the page
is generated, because another platform's libm can move a figure in its
third decimal. Sections 2 to 4 are left to the script: the size study alone
is 20,000 replications a cell across three distributions, 102 s of the
run.
