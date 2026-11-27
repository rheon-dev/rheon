<!-- kind: reference -->
# Models

The reference entry of each of the thirty-seven models a spec can name, by
family, for a reader who knows which model they want to look up. A
model's parameters, their defaults and its refusals are in its docstring,
and the fields it writes in [OUTPUTS.md](OUTPUTS.md).

**The update rules share one notation.** A symbol in this table means the
same thing in every entry its *used by* column names, and each entry defines its other
symbols beside its rule:

| symbol | what it is | used by |
|---|---|---|
| `z` | `[1, x]` when there is an intercept, and `x` when there is not | the linear models |
| `w`, `W` | the row's weight, and the weight accumulated before the row | every model |
| `λ` | the row's decay, `0.5 ** (Δclock / half_life)`, or `lam ** Δclock` when the spec gives `lam=`, with `Δclock` the clock's step since the last row that counted, one the model did not skip and of a weight above 0, capped at `gap_cap` (or `session_gap` at a session change) ([Time and decay](explanation/time-and-decay.md)) | every model that decays |
| `S_j`, `η_j` | target `j`'s running means of `z zᵀ` and of `z·y_j`, which the least-squares fits solve from | `ewridge`, `huber`, `quantile`, `ar` |
| `e` | the row's residual: `y` less the prediction made before the row | `huber`, `quantile` |
| `ω` | a ridge added to a covariance: `precision_prior`, fading as `ew_cov`'s prior does, so that a row's class or state can be predicted from the model's first row | `ew_class`, `hmm` under `"tied"` or `learn = False` |
| `P₀`, `ψ` | a stabilized prior a covariance forgets toward: `prior_rows` rows of weight at the features' pooled within-state variance `ψ` (at each state's given covariance, where `covs` is given), which does not fade | `hmm` under `"full"` and `"diag"` with `learn = True` |

Without a clock column, each row is one unit of clock. A model keeps its
running sums as means, decayed sums or centered second moments, and [The
running sums that
decay](explanation/time-and-decay.md#the-running-sums-that-decay) says
which models keep each, and why.

## The models at a glance

The thirty-seven models fall into four families, and each family answers one
kind of question about a stream. Each name links its builder's reference
page, which gives its keywords and their defaults, and *entry* links its
entry on this page. [Choose a model](how-to/choose-a-model.md) says which
to start with for which question.

| model | learns by | what it is |
|---|---|---|
| **[Linear models](#linear-models)** | | predict a numeric target, from features or from its own past |
| [`ewridge`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ewridge) · [entry](#ewridge--exponentially-weighted-ridge-regression) | solve | exponentially weighted ridge regression; several ridge values and feature sets are solved from the same running sums at almost no extra work |
| [`rls`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.rls) · [entry](#rls--recursive-least-squares) | solve | recursive least squares, in the numerically safe square-root form |
| [`lasso`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.lasso) · [entry](#lasso--lasso-path-its-penalty-chosen-as-it-runs) | solve | lasso and elastic-net path, with the penalty chosen as the stream runs |
| [`kalman`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.kalman) · [entry](#kalman--random-walk-β-dynamic-linear-model) | filter | a Kalman filter whose coefficients drift as random walks |
| [`huber`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.huber) · [entry](#huber--robust-regression) | reweight | Huber regression, which down-weights the rows a prediction misses by far |
| [`quantile`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.quantile) · [entry](#quantile--robust-regression) | reweight | quantile regression, the fit that a share `tau` of the targets fall below |
| [`expectile`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.expectile) · [entry](#expectile--expectile-regression) | reweight | expectile regression, the squared error with the rows above the fit weighed `tau` and those below `1 - tau`; exact over a window of rows or of the clock |
| [`sgd`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.sgd) · [entry](#sgd--stochastic-gradient-descent) | step | stochastic gradient descent with squared, Huber, quantile, ε-insensitive, Poisson and logistic losses |
| [`pa`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.pa) · [entry](#pa--passive-aggressive-regression) | step | passive-aggressive regression, with no learning rate to tune |
| [`ftrl`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ftrl) · [entry](#ftrl--online-logistic-regression) | step | FTRL-proximal regression, logistic by default, with an L1 penalty that zeroes coefficients |
| [`glm`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.glm) · [entry](#glm--generalized-linear-models-poisson-gamma-binomial-tweedie) | step | Poisson, Gamma, binomial and Tweedie regression, a Newton step a row on the likelihood, for counts, positive amounts, proportions and amounts with an exact zero |
| [`holt`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.holt) · [entry](#holt--holts-linear-trend) | step | Holt's linear trend, the baseline that uses no features |
| [`ar`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ar) · [entry](#ar--an-autoregression-on-the-targets-own-past) | solve | an autoregression for each target, AR(p), by least squares on lags the model keeps itself, with the features as further inputs, and `AutoReg`'s fit with no decay |
| [`isotonic`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.isotonic) · [entry](#isotonic--a-monotone-function-of-one-feature) | solve | isotonic regression of each target on one feature, the best non-decreasing (or non-increasing) function, refit by pool-adjacent-violators on every row: the monotone calibrator, scikit-learn's `IsotonicRegression` on the same rows |
| [`vector_ar`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.vector_ar) · [entry](#vector_ar--a-vector-autoregression-with-granger-tests) | solve | a vector autoregression of the targets together, VAR(p), with Granger causality tests on every row and impulse responses read from the bank; statsmodels' `VAR` with no decay |
| **[Moments and correlation](#moments-and-correlation)** | | track the running moments and correlations of the columns you name |
| [`ew_cov`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ew_cov) · [entry](#ew_cov--exponentially-weighted-moments) | accumulate | running mean, variance, covariance, correlation, partial correlation, Mahalanobis distance and principal components |
| [`pca`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.pca) · [entry](#pca--streaming-principal-components) | step | principal components learned a row at a time in state that grows with the components times the columns, not the columns squared: the incremental SVD, CCIPCA or Oja's rule, each component withheld until the data tells it apart from its neighbors |
| [`ewm_moments`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ewm_moments) · [entry](#ewm_moments--moments-to-any-order-lagged-or-not) | accumulate | moments and cross-moments of columns to any order, each leg perhaps lagged, each a term you name: skew, kurtosis, cumulants, co-skewness, autocovariances and autocorrelations, decayed or over a window, unbiased to order 4 and with standard errors |
| [`marginal`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.marginal) · [entry](#marginal--every-pairs-moments-kept-in-the-state) | accumulate | every (feature, target) pair's running mean, variance, covariance, correlation, slope and t, for a wide set of columns, kept in the state and read back as a table; optionally at a set of lags, and with binned target moments for the relations a correlation cannot see |
| [`deco`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.deco) · [entry](#deco--one-correlation-for-the-whole-matrix) | accumulate | one correlation for the whole matrix: Engle & Kelly's equicorrelation, or one per block and per pair of blocks |
| [`rcov`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.rcov) · [entry](#rcov--a-blocks-realized-covariance-robust-to-noise) | accumulate | a block's realized covariance, robust to microstructure noise: the Barndorff-Nielsen–Hansen–Lunde–Shephard kernel or Christensen–Kinnebrock–Podolskij pre-averaging, reported when a group closes |
| [`audit`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.audit) · [entry](#audit--what-a-streams-columns-hold) | accumulate | what each column holds, counted in one pass: nulls, NaNs and infinities apart, repeated values, runs, persistence, tails and duplicated columns, and the clock's duplicate stamps and gaps, read back as a table and by `bank.check()` |
| [`spectral`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.spectral) · [entry](#spectral--how-strong-is-each-cycle) | accumulate | the power of each target at chosen periods, from an exponentially weighted Fourier transform on the clock, read as scipy's periodogram |
| [`garch`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.garch) · [entry](#garch--how-large-will-the-next-return-be) | step | the GARCH family, a conditional variance for each return series: GARCH(1,1), GJR, EGARCH and FIGARCH, with normal, Student-t or skewed-t innovations, its parameters learned a row at a time after a start fitted by likelihood |
| [`dcc`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.dcc) · [entry](#dcc--how-do-the-correlations-move-from-row-to-row) | step | dynamic conditional correlation and its asymmetric form: the full correlation matrix of a few return series, each standardized by its own GARCH, moving every row, its two or three parameters given or learned a row at a time after a start fitted by quasi-likelihood |
| [`realized`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.realized) · [entry](#realized--how-much-did-it-move-and-did-it-jump) | accumulate | realized volatility and the jump tests on it, for each return series: realized variance, bipower and tripower variation, tripower quarticity, Barndorff-Nielsen and Shephard's ratio test, and Lee and Mykland's statistic of each return with its flag |
| [`power_variation`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.power_variation) · [entry](#power_variation--power-variation-and-kernel-weighted-autocovariance) | accumulate | power variation and kernel-weighted autocovariance sums, for each return series: named multipower terms, kernel sums (Bartlett, Parzen, Tukey-Hanning, rectangular; flat-top or centered) and subsampled scales, with their count of returns |
| [`tdigest`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.tdigest) · [entry](#tdigest--where-do-this-columns-quantiles-stand) | accumulate | quantiles of each column at the levels you list, from a merging t-digest whose error is uniform in rank, each level withheld until the rows identify it |
| [`ddsketch`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ddsketch) · [entry](#ddsketch--how-large-are-this-columns-quantiles-and-tails) | accumulate | quantiles and tail means of each column within a relative accuracy `alpha` of the value, from a DDSketch over every row or over a window of rows or of the clock |
| **[Clustering and classification](#clustering-and-classification)** | | put labels on rows |
| [`kmeans`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.kmeans) · [entry](#kmeans--exponentially-weighted-k-means) | step | exponentially weighted k-means: cluster labels assigned before the row is learned from, with a split–merge move that finds a cluster born after seeding |
| [`micro`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.micro) · [entry](#micro--density-based-clustering-any-shape) | step | density-based clustering: DenStream micro-clusters linked into clusters of any shape and number, flagging the rows that belong to none |
| [`ew_class`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ew_class) · [entry](#ew_class--gaussian-classification-on-ew_cov-moments) | accumulate | Gaussian classification, QDA, LDA or naive Bayes, with one set of running moments per class: a label column in, class probabilities out |
| **[Sequential tests and regimes](#sequential-tests-and-regimes)** | | give evidence that something holds or has changed, and which regime the stream is in |
| [`seqtest`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.seqtest) · [entry](#seqtest--a-sequential-test-of-a-sign-by-betting) | test | a sequential test of a sign by betting, giving evidence you can read at any row: on its own, of a column's sign; with `a` and `b`, of whether one spec of the bank predicts closer than another |
| [`corrchange`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.corrchange) · [entry](#corrchange--has-the-correlation-structure-changed) | test | has the correlation structure changed: the Wied–Krämer–Dehling constancy test span by span, the Wied–Galeano detector row by row against a stable history, or the size of a change between two windows against a permutation null |
| [`bocpd`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.bocpd) · [entry](#bocpd--how-long-has-this-regime-lasted) | filter | how long this regime has lasted: Adams & MacKay's run-length posterior, so the answer is the regime's age and not a flag |
| [`hmm`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.hmm) · [entry](#hmm--which-regime-are-we-in) | filter | a Gaussian hidden Markov model, filtered as the stream runs: `ew_class` without the labels, with a transition matrix that can be learned |
| [`hawkes`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.hawkes) · [entry](#hawkes--how-much-does-each-event-excite-the-next) | solve | a self-exciting point process: each node's event intensity before the row, and how much an event of each node adds to the next, fitted exactly by least squares over exponential kernels |

**How a model learns from a row decides whether, with decay off, its
answer depends on the row order:**

| learns by | what the model does with a row | with decay off | order-dependent even so |
|---|---|---|---|
| **solve** | keeps running sums and computes its coefficients from them | converges to the batch answer, in any row order | `lasso`'s `penalty_selected`, ranked by out-of-sample error, though its path converges in any order |
| **accumulate** | keeps running sums and reports them | converges to the batch answer, in any row order | a lag, in `ew_cov`, `marginal` or `ewm_moments`, which counts learned rows; `rcov`'s block and `deco`'s per-row estimate; `audit`'s runs, smallest change, lag pairs and clock steps, and past `distinct_cap` the digest's `median`, `mad` and `robust_z` and the counters' `top_value`, `top_count`, `second_count` and `equal_by_chance`, each within its bound |
| **reweight** | solves from running sums, but lets the fit before each row decide how that row enters them | depends on the row order | |
| **step** | moves its coefficients a little on each row | depends on the row order | |
| **filter** | carries a belief forward from row to row | depends on the row order | |
| **test** | counts evidence as it goes | depends on the row order | |

## What every spec takes

Every spec, whichever model it names, takes the parameters that say how
its model reads the stream, one chunk at a time. Most models take
all of them, and a model that refuses one says so when the spec is built.
With a `group`, every parameter applies within each group, the clock
included. The [`rheon.spec`](https://rheon-dev.github.io/rheon/spec.html)
reference gives each one's units and default.

### Parameters

| the question | the parameters | where it is described |
|---|---|---|
| which columns a model reads, and what its output column is called | the spec's name, `targets`, `features`, `fit_intercept` | [What a spec names](#what-a-spec-names) |
| how far apart two rows are, and how fast the past fades | `clock`, `half_life`, `lam`, `gap_cap` | [The clock and its units](#the-clock-and-its-units), and [Time and decay](explanation/time-and-decay.md) for why |
| what a session boundary, a gap and a step back do | `session`, `session_gap`, `gap_cap`, `restart_after_step_back` | [The clock's policy](#the-clocks-policy) |
| what a fit becomes when nothing fades | `half_life=inf`, `lam=1.0` | [Without a decay](explanation/time-and-decay.md#without-a-decay) |
| when an old row stops counting at all | `window_size`, in the models [A hard window](#a-hard-window) names | [A hard window](#a-hard-window), and [Forget rows past a window](how-to/forget-rows-past-a-window.md) |
| how to fit along a feature instead of time | `clock` | [Fit locally along a feature](how-to/fit-locally-along-a-feature.md) |
| which row order a bank needs, and how it checks it | | [Put rows in clock order](how-to/put-rows-in-clock-order.md), and [why you set the order](explanation/chunk-invariance-and-streaming.md#why-you-set-the-row-order) |
| which rows belong to which model | `group`, `group_close` | [Fit one model per group](how-to/fit-one-model-per-group.md) and [Close groups and read each one's row](how-to/close-groups.md) |
| how much each row counts | `weight` | [Weight rows](how-to/weight-rows.md) |
| when a model has seen enough to report | `min_weight`, `min_settled_frac`, `max_error_inflation` | [Warm-up gates](#warm-up-gates) |
| when a target may be learned | `embargo`, `label_clock`, `max_held_rows`, `held_overflow` | [Learn targets known late](how-to/learn-targets-known-late.md) |
| what a null does, and how to keep a model from learning a row | a null, a weight of 0, `predict` | [Hold a row back](how-to/hold-a-row-back.md) |
| which coefficients and diagnostics each row writes | `coef_every`, `max_rows_between_coefs`, the `emit_*` switches and their half-lives | [OUTPUTS.md](OUTPUTS.md) and [DIAGNOSTICS.md](DIAGNOSTICS.md) |

#### What a spec names

A spec describes one model: which model it is, the columns it reads, and
how it treats time. The name of the spec (the first parameter) is used to
name the output column: `rheon.spec.ewridge("ridge", ...)` adds a column named
`ridge`. Each row of that column is a record of the model's fields for the
row, and the input's columns pass through untouched.
This code uses `df` from [Example data](EXAMPLE-DATA.md):

```python
ridge_spec = rp.spec.ewridge(
    "ridge",
    targets=["y"],                 # at least one; the targets of one spec share the features' running sums
    features=["x0", "x1", "x2"],   # numeric columns of any width, Decimal and Boolean included; read as 64-bit floats
    fit_intercept=True,            # the default: the fit has a level of its own
    clock="t", half_life=600.0, gap_cap=300.0,   # how it treats time: see The clock and its units
)
fitted = rp.ModelBank([ridge_spec]).fit_predict(df)
print(len(df.columns), "columns in, then", fitted.columns[len(df.columns):])
print("each row of ridge holds", fitted["ridge"].struct.fields)
```

```text
13 columns in, then ['ridge']
each row of ridge holds ['pred_y', 'resid_y', 'weight_sum', 'settled_frac', 'withheld_reason', 'coef', 'support_coef']
```

A target can also be computed from the rows after each row, as a formula
in the spec's `targets`: [Make a target that looks
ahead](how-to/make-a-target-that-looks-ahead.md) shows how, and how to
write a ratio target that sits about zero.

#### The clock and its units

**Write every clock parameter in the clock column's own kind of value,
since a timestamp carries its unit and a number does not.**

| the clock | a clock parameter is | for example |
|---|---|---|
| a `Datetime`, `Date` or `Duration` column: a *temporal* clock | a duration | `half_life=pl.duration(minutes=10)` |
| a numeric column, such as seconds or cumulative traded volume | a number of the column's own units | `half_life=600.0` |
| none | a number of rows | `half_life=100`: a row 100 rows back counts half as much as the latest |

A duration is written three ways: `pl.duration(minutes=10)`,
`timedelta(minutes=10)`, or Polars' duration text `"10m"`. `lam`, the
weight kept per clock unit, may stand in for `half_life` on a numeric
clock or with no clock.

**Every clock parameter acts on the clock, so it means the same span of
time however densely the rows arrive.** The clock parameters are
`half_life`, `gap_cap`, `restart_after_step_back`, `session_gap`, `embargo`,
`drift_threshold` and `coef_every`, and a model's `window_size`, its cadences
(`snapshot_every`, `solve_every`, `pca_every`, `prune_every`) and its own
half-lives. A few things still count rows, such as `warm_rows` and the
caps named `max_rows_between_*`, and each model's docstring names its own.

**A count of rows takes a clock form beside it, wherever the count has a
meaning on the clock.** The warm-ups are `warm_span`, beside `kmeans`',
`hmm`'s and `bocpd`'s `warm_rows`. The first rows' refits are `refit_span`
beside `glm`'s and `expectile`'s `refit_rows`, and `huber_refit_span` and
`quantile_refit_span` beside theirs. The cadences are `kmeans`'
`update_every` and `split_merge_every`, and `corrchange`'s `permute_every`.
Each takes a number of the clock's units or a duration, and is refused
beside its count. A start or a refit over a span holds a target's rows
from its first learned row through the first row at least the span after
it, read off the rows' exact stamps; a cadence fires every span since its
last event. A row of weight 0 is in no span. On a clock of one unit a row,
a span of `D` is the count `D + 1` for a start and `D` for a cadence, to
the bit. The data decides how many rows a span holds, so its rows are held
to a cap as they come. Past half of it comes a warning, once a group, and
the row that would pass it stops the run with an error naming the group
and the bytes. A count that is a lag of a series of observations, a
detector's count of observations or a bound on the state has no clock
form, and its docstring says so.

**Quantities measured on a temporal clock reach the output in seconds.**
`holt`'s trend, for one, is per second. The clocks themselves keep the
clock column's own type: `emit_clocks` ([Learn targets known
late](how-to/learn-targets-known-late.md#check-that-it-worked)), and the clock columns of `summary()`,
`groups()` and `closed_groups()`. The recipe [Two clocks, one
rate](#two-clocks-one-rate) shows both.

**A temporal clock is read in its own integer nanoseconds, and an integer
clock as its integers**, so the step between two rows is exact. [Decay along
the clock](explanation/time-and-decay.md#decay-along-the-clock)
says why, and what range of dates a temporal clock covers.

#### The clock's policy

A stream from a market has session boundaries, gaps and steps back in its
clock. The spec has a parameter for each, and together they are the
clock's *policy*. [Handle sessions, gaps and steps
back](how-to/handle-sessions-gaps-and-steps-back.md) shows each one's
effect on a fit:

| the stream has | the parameter | what it does |
|---|---|---|
| a **session** boundary, such as the start of a trading day, where the clock's jump is not elapsed time | `session`, a column whose value changes at a session boundary, and `session_gap`, required with it unless `group_close="session"` closes the group at the change instead ([Close groups and read each one's row](how-to/close-groups.md)) | applies a step you choose, `session_gap`, in place of the jump, capped at `gap_cap` like any step. `session_gap="reset"` starts the model over instead |
| a **gap** in the clock, such as an hour with no rows | `gap_cap`, required with a clock, finite and above 0 | caps each step between two rows a model learns from, so a quiet hour ages the fit no more than `gap_cap` would. After a run of skipped rows, the next row's step is capped too, however long the run |
| a clock that **steps back**, such as a replayed day | `restart_after_step_back` | resets the model at a step back larger than the setting, and refuses one no larger as a late row. Unset, the default, every step back is refused ([Put rows in clock order](how-to/put-rows-in-clock-order.md)) |

**A stream has a *clock break* wherever `gap_cap` shortened a step, and
wherever the session changes.** A model that keeps past rows by position,
as a lag does, empties them at a clock break, where the previous row no
longer stands for the moment just before.

#### A hard window

`window_size` makes every row older than a span count for nothing ([Forget
rows past a window](how-to/forget-rows-past-a-window.md) shows how, and
what each model keeps inside the window). `ewridge`, `lasso`, `ar`,
`vector_ar`, `ew_cov`, `ew_class`, `marginal` and `spectral` cut it from snapshots of
their running sums, spaced by `snapshot_every` and
`max_rows_between_snapshots` and held under `window_budget`. `expectile`,
`isotonic`, `hawkes`, `realized`, `power_variation`, `ddsketch`, `ewm_moments` and
`corrchange` take a `window_size` of their own, without snapshots, and
each one's entry says what it keeps and how its edge is read. Five things
to know before reading the numbers of a model under a snapshot window:

| | |
|---|---|
| the clock is the **decayed** one | a row's age adds up each step after `gap_cap` and any `session_gap` |
| the edge is a **discontinuity** | a row aging out drops its whole weight at once, so the series has small steps that a plain exponential average does not |
| it is a **subtraction** | precision falls with the fraction discarded: negligible at a window of three half-lives, worse as the window shortens toward the half-life |
| `weight_sum` is the weight inside the window | so `min_weight` gates on a weight that stops rising once the window is full |
| a long step **empties** the window | a step longer than `window_size`, after `gap_cap`, empties the window, and the model reports nulls from the row after it, never stale numbers. A cap shorter than the window never empties it |

#### Warm-up gates

A model should not report a number it is not yet informed enough to give.
Every model with a decay shares two gates that say what "informed enough"
means, and four linear models (`ewridge`, `rls`, `lasso` and `kalman`) add
a third, on by default on `ewridge` alone.
The gates hold back output only: the model learns from every row either
way. [Warm-up and readiness](explanation/warm-up-and-readiness.md)
explains what each gate measures, and why, and [Decide when a model is
ready](how-to/decide-when-a-model-is-ready.md) shows how to choose one.

| setting | default | withholds a prediction while | read from |
|---|---|---|---|
| `min_weight` | by model, as the table of defaults gives | `weight_sum` is below it | the weight behind the state |
| `min_settled_frac` | `0`, off | the model has come less than this fraction of the way toward its steady state under decay | `settled_frac = 1 − 2^(−T / half_life)`, with `T` the decay time the model has seen: `0.5` at one half-life, `0.75` at two, whatever the row rate |
| `max_error_inflation` | `sqrt(2)` on `ewridge`; off on `rls`, `kalman` and `lasso` | estimation error would inflate the prediction's error over the noise floor by more than this factor | `error_inflation = sqrt(1 + edf / n_kish)`: the effective degrees of freedom the fit used, over Kish's effective sample size behind it |

**`weight_sum` means the same in every model, so one `min_weight` means
the same for every spec in a bank.** It is the *weight behind the state*
that produced a row's prediction: the weight as the previous row left it,
before this row's decay and before the model learns the row, and `0` on a stream's first row
([`weight_sum`: the weight behind a
prediction](explanation/warm-up-and-readiness.md#weight_sum-the-weight-behind-a-prediction)).

**Every prediction is null until the model's weight reaches `min_weight`, and so are most fields read from the fit ([OUTPUTS.md](OUTPUTS.md) lists which).** The readiness fields, such as `weight_sum` and `withheld_reason`, are filled. A
regression model checks each target against that target's own weight, the
weight of the rows it was present on, so an often-null target reports
later than the others. For `rls`, which learns a row only when every target
is present, that is the weight of the rows it learned from. The
`weight_sum` field is the shared weight either way. To set one threshold
per target, give `min_weight` a list.

**`emit_target_weight=True` writes the weight each target's gate reads, as
`weight_<target>` beside `weight_sum`.** A row whose target is null teaches
that target nothing, so it adds to `weight_sum` and not to that target's
weight. A row can then show `weight_sum` above `min_weight` together with
`withheld_reason = "below_min_weight"`, and `weight_<target>` is the number
below the floor. It has `weight_sum`'s meaning: the weight before the row and
before its own decay, inside the window under one. Every model with a target
keeps a weight per target, except `seqtest`, `hawkes` and `dcc`. Those three
gate every target on the shared weight, so their `weight_<target>` is
`weight_sum`. A model with no target, and `marginal`, which writes no
prediction to withhold, refuse the switch. To keep a row out of the learning
without dropping it, give it weight 0 in the `weight` column, which is the
library's learn mask: the row is scored and teaches nothing. The default depends on the model:

| default | models |
|---|---|
| one per unknown: the features, and the intercept when there is one | `lasso`, `kalman`, `huber`, `quantile`, `expectile`, `rls`, `sgd`, `pa`, `ftrl`, `glm` |
| the feature count plus one (1 for `holt`) | `ew_cov`, `ew_class`, `kmeans`, `micro`, `holt` |
| 3 | `marginal`, `deco` |
| 1 | `bocpd` |
| 0, each having a gate of its own or nothing to gate | `ewridge`, `ewm_moments`, `seqtest`, `rcov`, `hmm`, `corrchange`, `audit`, `hawkes`, `garch`, `dcc`, `realized`, `pca`, `power_variation`, `tdigest`, `ddsketch`, `ar`, `spectral`, `isotonic`, `vector_ar` |

**`min_settled_frac` is off by default.** Set it when the half-life was
chosen to average over regimes or seasons that a shorter history would not
represent ([`settled_frac`: how near the steady
state](explanation/warm-up-and-readiness.md#settled_frac-how-near-the-steady-state)
gives the reason).

**`max_error_inflation` follows the size of the fit:** add a feature, and
the gate holds predictions back longer. Each model that has it reads its
own estimation variance ([The noise
gate](explanation/warm-up-and-readiness.md#the-noise-gate-estimation-error-against-the-noise)
derives it). The others refuse the setting and say why:

| model | `error_inflation` | |
|---|---|---|
| `ewridge` | `sqrt(1 + edf / n_kish)` | on by default at `sqrt(2)` |
| `rls` | `sqrt(1 + k / n_kish)`, `k` the coefficients: its `edf` is at most `k` under the fading prior | off unless set |
| `lasso` | `sqrt(1 + df / n_kish)` per penalty, `df` the active coefficients and the intercept | off unless set |
| `kalman` | `sqrt(1 + z' P z / R)` for each row: the prediction's own variance, its coefficients' covariance carried through the row's clock gap, over the noise | off unless set; read per row, so a row far from the design is held back where others pass |

**Each row says how ready its model was, and why a prediction is
missing.** `withheld_reason` names the gate that held a row back, and the
gates are checked in the order of its values, so the first that applies is
the one named. The recipe [Three gates on one
spec](#three-gates-on-one-spec) counts the rows each withholds. The
readiness fields:

| field | on | what it holds |
|---|---|---|
| `settled_frac` | every model but `marginal`, `rcov` and `audit` | how far the model has come toward its steady state under decay |
| `withheld_reason` | every model but `marginal`, `rcov` and `audit` | why `pred_y` is null, or null when it is not: `below_min_settled_frac`, `below_min_weight`, `above_max_error_inflation`, `below_warm_rows` (the rows `kmeans`, `hmm` and `bocpd` hold to seed themselves from, the rows `garch` buffers for a target's start, the rows a `holt` heuristic start holds, the rows before `kalman` has sized any of a target's coefficients, standardized with no intercept, and the rows before `deco`'s standardizer holds more than five effective rows, while it has no level), `ill_conditioned` (an `ewridge` fit whose normal equations are too nearly singular to keep a digit), `unstable_rate` (an `sgd` target whose learning rate is past its stability bound, so the fit diverges), `not_identified` (a `garch` start whose rows do not tell its variance process from a constant variance, or an `isotonic` fit read at a feature beyond the rows before it), `scale_ratio_moved` (a `corrchange` flag withheld on a span whose two columns' scale ratio moved), `not_separated` (a `pca` component the data does not yet tell apart from a neighbor), `level_beyond_sample` (a `tdigest` or `ddsketch` level past what the rows before it identify) `input_negative` (a `garch` row under GARCH or GJR whose input is below 0) or `target_not_positive` (a multiplicative `holt` season's row predicted and not learned, its target or its update 0 or below: the one reason beside a prediction, and in that form's `enum` alone), in that order of precedence |
| `error_inflation_y` | `ewridge`, `rls`, `kalman`, under `emit_error_inflation` | the row's own estimation variance over the noise: high on a row leaning on a direction the data never showed, at one triangular solve a row |
| `support_coef` | `ewridge`, `huber`, `quantile`, beside `coef` | each coefficient's data share, `1 - ridge * (S^-1)_jj` on the system the solve inverts. A duplicated pair of features reads 0.5 each |
| `se_coef` | `ewridge`, `rls`, `kalman`, under `emit_se_coef`, beside `coef` | each coefficient's standard error in its own units: `sigma` times the fit's own covariance, `kalman`'s posterior exactly. A report, not a gate; it errs large in warm-up and under a ridge |

**`summary()` carries the same readings per group, and the calls that
learn warn of a gate that will not open.** `summary()` adds
`weight_sum_settled`, the weight `weight_sum` tops out at. Once
`settled_frac` reaches 0.95, `fit`, `fit_predict`, `fit_predict_batches`
and a query's `rheon.fit_predict` issue a `ReadinessWarning` that names,
once, a coefficient more ridge than data. It also names a noise gate or a
`min_weight` that has withheld every row for a further half-life, and that
the rows, at their rate so far, would not meet. It gives the half-life or
the ceiling that would change that, and promises nothing of rows that come
faster. The command line closes a run with one line per spec whose groups
ended withheld or with a coefficient more ridge than data.

### What the bank refuses

The bank checks every spec against the input's columns, and every clock
parameter against the clock's kind, before it learns any row. A value that
only a row can show, such as a negative weight or a null clock value, is
refused when the chunk that holds it arrives. Each error names the spec or the column it concerns, and the remedy is its last sentence.

| refused | because |
|---|---|
| a spec named like an input column | its output column would replace that column |
| a text column in `targets` or `features` | a model reads numbers: convert it with Polars' `with_columns` in the query before the bank, `cast(pl.Float64)` for numbers stored as text and a comparison such as `pl.col("venue") == "X"` for a category ([Refused specs](#refused-specs)) |
| a column the input does not have | it names the column, the spec that wanted it and the role it had there |
| a plain number as a clock parameter on a `Datetime` clock | it would silently take the column's storage unit: `half_life=600` on a microsecond column would mean 600 microseconds |
| a duration as a clock parameter on a numeric clock | it has nothing to measure it against |
| a spec that mixes the two kinds | |
| a duration finer than the clock can act on, such as `gap_cap="12h"` on a `Date` clock, which moves in days | |
| `lam`, a decay per clock unit, on a temporal clock | it has no duration form, so a temporal clock takes `half_life` instead |
| a negative weight | a weight of 0 is legal, and the model learns nothing from its row ([Weight rows](how-to/weight-rows.md)) |

Only `0` and `inf`, which mean the same in every unit, may stay numbers
where a clock parameter takes them.

**Catch a refusal by the kind of problem: `TypeError` for a value of the
wrong type, `ValueError` for a value out of range or a column the input
lacks.** A column the input lacks raises `rheon.ColumnNotFoundError`, a
subclass of Polars' own `ColumnNotFoundError` and of `ValueError`. A column
or value of the wrong type, such as a text feature, raises
`rheon.WrongTypeError`, a `TypeError`. A refused value raises `ValueError`, or
`rheon.RefusedValueError`, a `ValueError`.

**A clock column the bank cannot read exactly is refused, by row or by
name.** A `UInt64` value past the largest `Int64` is refused by row. An
`Int128` or `UInt128` column is refused by name: cast it to `Int64` if its
values fit, or after subtracting an origin. A `UInt128` column is refused
wherever it is in the frame, since no frame carrying one can reach the
bank. A later chunk whose clock is a float, or an integer of another width,
is refused by name. A null, NaN or infinite clock value is refused by row.
A feature may be null and its row skipped, but a clock value cannot: every
later row's step is measured from it.

### Examples

Each recipe runs on the [example data](EXAMPLE-DATA.md) and prints what it shows.

#### Refused specs

This code prints two refusals, then fits the venue as a comparison. Its
last line finds the truth planted in `y`: an intercept of 1, 2 on `x0`, −1
on `x1`, and nothing on the venue.
It reads `df` and `lf` from [Example data](EXAMPLE-DATA.md):

```python
# Two specs the bank refuses, each refusal printed.
for refused in [
    rp.spec.ewridge("x0", targets=["y"], features=["x1"], half_life=600.0),          # named like a column
    rp.spec.ewridge("by_venue", targets=["y"], features=["venue"], half_life=600.0),  # a text feature
]:
    try:
        rp.ModelBank([refused]).fit_predict(df)
    except (TypeError, ValueError) as err:   # a text feature is a TypeError
        print(err)

# The venue fitted as a comparison.
by_venue = rp.spec.ewridge("by_venue", targets=["y"], features=["x0", "x1", "is_x"], half_life=float("inf"))
coded = lf.with_columns(is_x=pl.col("venue") == "X").rheon.fit_predict([by_venue]).collect()
print("intercept, x0, x1, venue X:", [round(c, 2) for c in coded["by_venue"].struct.field("coef")[-1]])
```

```text
spec "x0": it has the same name as an input column, and its output column would replace that column. Rename the spec.
spec "by_venue": feature column "venue" must be a number, but it is String. Cast it, such as with pl.col("venue").cast(pl.Float64).
intercept, x0, x1, venue X: [1.02, 1.99, -1.01, -0.05]
```

#### Two clocks, one rate

This code fits `holt`, a level and its trend, to `t`, a column that rises
by one each minute, once on the numeric clock `t` and once on the
`Datetime` clock `ts`. The two clocks mark the same rows, so the weights
agree, and the trend is the same rate in two units. Its last lines show
the bank refusing a plain number on the `Datetime` clock.
It reads `df` from [Example data](EXAMPLE-DATA.md):

```python
import textwrap

# The same model on each clock, read on the last row.
on_t = rp.spec.holt("on_t", targets=["t"], clock="t", half_life=100.0, gap_cap=300.0)       # numbers: t counts minutes
on_ts = rp.spec.holt("on_ts", targets=["t"], clock="ts", half_life="100m", gap_cap="300m")  # durations: ts is a Datetime
last = rp.ModelBank([on_t, on_ts]).fit_predict(df).tail(1)
for name, unit in [("on_t", "per unit of t, a minute"), ("on_ts", "per second")]:
    fields = last[name].struct
    print(f"{name}: weight_sum {fields.field('weight_sum')[0]:.2f}, trend {fields.field('coef')[0][1]:.6f} {unit}")

# The same spec on ts with plain numbers: refused.
try:
    rp.ModelBank([rp.spec.holt("wrong", targets=["t"], clock="ts", half_life=100.0, gap_cap=300.0)]).fit_predict(df)
except ValueError as err:   # a plain number on a Datetime clock
    print(textwrap.fill(str(err), 100))
```

```text
on_t: weight_sum 135.66, trend 1.000000 per unit of t, a minute
on_ts: weight_sum 135.66, trend 0.016667 per second
spec "wrong": clock column "ts" is Datetime("us"), a temporal clock, but half_life is a plain
number. A number has no unit, so the unit the column is stored in, such as microseconds since 1970,
would silently become its unit. Give half_life and the other clock parameters as durations, such as
"10m" or pl.duration(minutes=10), or cast the clock to a number of the units you mean, such as with
pl.col("ts").dt.epoch("s").cast(pl.Float64) (pl.col("ts").dt.total_seconds() for a Duration).
```

#### Three gates on one spec

This code sets all three warm-up gates on one spec and counts the rows
each withholds.
It reads `df` from [Example data](EXAMPLE-DATA.md):

```python
warm = rp.spec.ewridge(
    "warm", targets=["y"], features=["x0", "x1"], clock="t", half_life=100.0, gap_cap=300.0,
    min_settled_frac=0.02,      # withhold until 2% of the way to the steady state under decay
    min_weight=8.0,             # and until the weight behind the fit reaches 8
    max_error_inflation=1.05,   # and while estimation error would add more than 5% over the noise floor
)
# How many rows each gate withheld, and the first row predicted.
rows = rp.ModelBank([warm]).fit_predict(df).unnest("warm")
counts = rows.group_by("withheld_reason").len().sort("withheld_reason", nulls_last=True)
for reason, n in counts.iter_rows():
    print(f"{reason or 'predicted'}: {n} rows")
first = rows.filter(pl.col("pred_y").is_not_null()).row(0, named=True)
print(f"first prediction at t = {first['t']:g}: weight_sum {first['weight_sum']:.2f}, settled_frac {first['settled_frac']:.3f}")
```

```text
below_min_settled_frac: 4 rows
below_min_weight: 5 rows
above_max_error_inflation: 21 rows
predicted: 370 rows
first prediction at t = 30: weight_sum 27.18, settled_frac 0.182
```

## Linear models

Fourteen models predict a numeric target. Eleven regress it on features,
`holt`, a baseline, uses none, `ar` regresses it on its own last values,
and `vector_ar` regresses every target on all of their last values, each
on features where they are given. Several of them take the same parameters:

| parameters | taken by | explained under |
|---|---|---|
| `ridge` | `ewridge`, where a list fits a grid; `huber`, `quantile`, `expectile`, `glm` and `ar`, as one number. `rls` refuses it and takes `delta` | [`ewridge`](#ewridge--exponentially-weighted-ridge-regression) and [`rls`](#rls--recursive-least-squares) |
| `solve_every`, `max_rows_between_solves` | `ewridge`, `lasso`, `huber`, `quantile`, `vector_ar` | [`ewridge`](#ewridge--exponentially-weighted-ridge-regression) |
| `target_gaps` | `ewridge`, `lasso` | [`ewridge`'s docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ewridge) |
| `window_size` | `ewridge`, `lasso`, `ar`, `expectile`, `vector_ar` | [A hard window](#a-hard-window), and [Forget rows past a window](how-to/forget-rows-past-a-window.md) |
| `standardize` | `ewridge`, `huber`, `quantile` and `ftrl`, off by default; `kalman`, `sgd` and `pa`, on by default. `lasso` always standardizes. `rls`, `expectile` and `glm` take no `standardize` and refuse it by name: `rls`'s prior (`delta`, `coef_prior`) is in the features' own units; and `expectile`'s and `glm`'s `ridge` is a penalty in the features' squared units. Scale such a feature before the bank | [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), [`kalman`](#kalman--random-walk-β-dynamic-linear-model) and [`ftrl`](#ftrl--online-logistic-regression) |
| `coef_min`, `coef_max`, `coef_sum` | `sgd`, `pa` | [`sgd`](#sgd--stochastic-gradient-descent) |
| `weight` | every model, though `rls`, `kalman`, `sgd`, `pa`, `ftrl` and `glm` read it differently from a weighted mean | [Weights](how-to/weight-rows.md) |

What `half_life` decays differs by model:

| model | `half_life` decays |
|---|---|
| `ewridge` | the running sums the fit is solved from |
| `lasso` | the same sums, and the error that picks the penalty unless `select_half_life` is given |
| `rls` | `A` and `b`, which it keeps as decayed sums |
| `kalman` | the observation-noise estimate and the features' standardization; `coef_half_life` sets how fast a coefficient may drift |
| `huber`, `quantile` | their running sums, kept per target, and the residuals' spread `σ` (for `huber`, at `huber_scale_half_life` when given) |
| `sgd` | `weight_sum`, the running moments `standardize` scales by, the spreads its cut and tube are drawn in (the Huber cut's at `huber_scale_half_life` when given), and adagrad's sums, but not the coefficients |
| `pa` | `weight_sum`, each target's own weight, the running moments `standardize` scales by and the spread its tube is drawn in, but not the coefficients |
| `ftrl` | its per-coordinate sums, but not its penalties, so a shorter half-life shrinks the fit more |
| `glm` | each target's Gram `G`, but not its ridge: the step's penalty ages toward 0 as the first row does and pulls toward the fits before, so a row of weight 0 moves no coefficient |
| `expectile` | each target's running sums, which its asymmetric fit is solved from |
| `holt` | the level, and the trend at `trend_half_life`, four times `half_life` by default |
| `ar` | each target's running sums, which its fit is solved from; not its lags, which count learned rows |
| `vector_ar` | the running sums its fit, its tests and its reads are solved from; not its lags, which count learned rows |

### `ewridge` — exponentially weighted ridge regression

*API:* [`rheon.spec.ewridge`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ewridge) — *Rust:* [`ewridge.rs`](../crates/rheon-core/src/ewridge.rs) — *Outputs:* [fields](OUTPUTS.md#ewridge)

Exponentially weighted ridge regression on running sums.

Start with `ewridge` for a numeric target, where its
[Gram](GLOSSARY.md#fitting-a-model), the sums it keeps of its columns'
means and co-moments, fits a grid of ridge values and feature sets for one
extra solve each. Change to another linear model when your data need what
it adds:
[`kalman`](#kalman--random-walk-β-dynamic-linear-model) for coefficients
that drift, [`huber`](#huber--robust-regression) for outliers or
[`quantile`](#quantile--robust-regression) for a quantile, and
[`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs) for a sparse
fit. Use [`sgd`](#sgd--stochastic-gradient-descent) or
[`pa`](#pa--passive-aggressive-regression) where learning a row must take O(k) time for
`k` features: this model takes O(k²) time.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ewridge),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#ewridge). The
parameters every spec takes are in [What every spec
takes](#what-every-spec-takes).

#### The update

Under the default `target_gaps="own_rows"`, `S_j` and `η_j` are taken over
the rows where `y_j` is present, and `W_j` is the
weight behind them. Under `"pairwise"`, one `S`, weighted by `W`, is taken
over every row, while `η_j`, `W_j` and the feature means the intercept is
centered on stay over `y_j`'s rows, and the slopes solve the centered
system the docstring's Notes give under *Target gaps*. `W` is the
`weight_sum` a row reports. The equations are the default's, and the
coefficients are solved on a schedule:

```
W'   = λW + w                       weight_sum, over every row
W_j' = λW_j + w                     on the rows where y_j is present, λW_j on the others
S_j' = (λW_j·S_j + w·z zᵀ) / W_j'   η_j' = (λW_j·η_j + w·z·y_j) / W_j'
solve, ridge_scale="mean":  (S_j + ridge·D) β_j = η_j + ridge·D·β₀      D = I with a 0 at the intercept
solve, ridge_scale="sum":   (W_j·S_j + p·ridge·I) β_j = W_j·η_j + p·ridge·β₀
                            p the product of every decay so far, β₀ = coef_prior (0 by default)
```

#### Example

**A grid of half-lives and ridge values is fitted in one pass.** Each
half-life keeps a set of running sums, and each ridge value is one more
solve from that set. Here a slope drifts from 1 to 3 over 20,000 rows, and
four fits read it at the last row:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# A slope on x that drifts from 1 to 3 over 20,000 rows, one row a clock unit.
rng = np.random.default_rng(1)
n = 20_000
x = rng.standard_normal(n)
slope = 1.0 + 2.0 * np.arange(n) / n
y = slope * x + 0.5 * rng.standard_normal(n)
pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y}).write_parquet("drift.parquet")

# One spec holds the whole grid, fitted in one pass.
drift = rp.spec.ewridge(
    "drift", targets=["y"], features=["x"], clock="t", gap_cap=10.0,
    half_life=[200.0, 20_000.0],   # a grid of two half-lives, one set of running sums each ...
    ridge=[1e-6, 0.1],             # ... by two ridges, solved from each set: four fits
)
bank = rp.ModelBank([drift])
bank.fit(pl.scan_parquet("drift.parquet"))   # learn every row, keep no output
# The truth at the last row, beside each fit's slope.
print(f"the true slope at the last row: {slope[-1]:.2f}")
for r in bank.coef().filter(pl.col("term") == "x").iter_rows(named=True):
    print(f"half-life {r['instance']}, ridge {r['ridge']:g}: slope {r['coef']:.2f}")
```

```text
the true slope at the last row: 3.00
half-life @h200, ridge 1e-06: slope 2.98
half-life @h200, ridge 0.1: slope 2.71
half-life @h20000, ridge 1e-06: slope 2.11
half-life @h20000, ridge 0.1: slope 1.92
```

The half-life of 200 rows follows the drift. The half-life of 20,000 still
weighs the rows from the middle of the stream, where the slope was near 2.
A ridge of 0.1 divides a slope by about `1 + 0.1/var(x)`, here 1.1, because
the ridge is added to the feature's variance in the solve.

#### See also

The guides to its settings:

- To fit a grid of `ridge` values and `feature_sets` from one Gram, see
  [Grids, instances and slots](explanation/grids-and-instances.md#how-a-grid-shares-its-work).
- To choose how often `ewridge` solves, with `solve_every` and
  `max_rows_between_solves`, see [Choose when `ewridge`
  solves](how-to/choose-when-ewridge-solves.md).
- To start from yesterday's fit, with `coef_prior` and `ridge_scale="sum"`,
  see [Start from a prior fit](how-to/start-from-a-prior-fit.md).
- To start each session from a blend with the long run, with
  `session_shrink` and `long_half_life`, see [Blend each session with the
  long run](how-to/blend-each-session-with-the-long-run.md).
- To fit hundreds of features fast, with `gram_block_rows` and
  `gram_threads`, see [Fit hundreds of
  features](how-to/fit-hundreds-of-features.md).

When `ewridge`'s fit is not the one you need, or to use its sums outside a
bank:

- [`rls`](#rls--recursive-least-squares), the same fit by recursive least
  squares, which loses half as many digits on a nearly collinear design.
- [`huber`](#huber--robust-regression), which keeps sums of the
  same form, each row down-weighted by its residual.
- [GRAM.md](GRAM.md), to read the sums back, pool them and solve them
  offline.

#### Notes

**With decay off and `ridge=0`, `ewridge` fits ordinary least squares over
every row seen, in any row order.** Its coefficients match
`numpy.linalg.lstsq` to 2e-13, fed forwards or backwards. On 6M rows × 20
features from a parquet stream, memory peaked at 1.4 GB, against 3.97 GB
for `lstsq` on the same rows.

**NIST's statistical reference datasets show where the solve stops
promising a digit.** Their hardest designs that double precision can solve
by the normal equations have correlation matrices conditioned at `2.7e6`,
far inside the limit at which `ewridge` withholds a fit. `Filip`, a
degree-10 polynomial at `1.5e16`, is past it, and is withheld. `rls`,
which rotates the design rather than squaring it, keeps 7 of `Filip`'s
digits. The docstring's Notes give the limit, and how each solve is
checked against it.

### `rls` — recursive least squares

*API:* [`rheon.spec.rls`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.rls) — *Rust:* [`rls.rs`](../crates/rheon-core/src/rls.rs) — *Outputs:* [fields](OUTPUTS.md#rls)

Recursive least squares, which solves the ridge fit exactly on every row.

Use `rls` when the coefficients must be current at every row: it learns
each row in O(k²) time for `k` features, by rotating the row into a kept
factor, where `ewridge` solves a `k × k` system on a schedule. Its prior
fades as rows arrive. Use
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) for a long
stream whose features can hold one value for a long time, which winds this
fit up, and [`huber`](#huber--robust-regression) with
`ridge_scale="sum"` for this fit under a Huber loss, solved on a schedule.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.rls),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#rls). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes).

#### The update

`A` and `b_j` are decayed sums of `z zᵀ` and `y_j·z`, not means. They start
from `A₀ = delta·I`, the classic RLS prior `P₀ = I/delta`.
Because `A` is a sum, a row's weight is on the sum scale, and a heavier
stream outweighs the starting prior sooner
([Weights](how-to/weight-rows.md)). The model keeps the Cholesky factor of
`A` and folds each row into it by Givens rotations (the square-root form):

```
A'   = λA + w·z zᵀ        b_j' = λb_j + w·y_j·z        β_j = A⁻¹ b_j
A₀   = delta·I            b_j₀ = delta·β₀_j             β₀_j = coef_prior (0 by default)
```

#### Example

**`rls` is `ewridge(ridge_scale="sum")` solved on every row, to better
than 1e-9.** Its `delta` is that `ridge`. Here the two run side by side on
a stream whose slope drifts:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# y = 0.5 + slope·x0 − x1 plus noise, with x0's slope drifting from 1 to 2.
rng = np.random.default_rng(2)
n = 5_000
x0, x1 = rng.standard_normal((2, n))
y = 0.5 + (1.0 + np.arange(n) / n) * x0 - x1 + 0.3 * rng.standard_normal(n)
pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": y}).write_parquet("rls.parquet")

# rls beside the ewridge it equals, on the same rows.
common = dict(targets=["y"], features=["x0", "x1"], clock="t", gap_cap=10.0, half_life=300.0)
specs = [
    rp.spec.rls("rls", delta=2.0, **common),                      # A starts at 2·I, a prior that fades
    rp.spec.ewridge("sum", ridge=2.0, ridge_scale="sum",          # the same prior on ewridge's sum scale ...
                    max_rows_between_solves=1, **common),         # ... solved on every row
]
out = pl.scan_parquet("rls.parquet").rheon.fit_predict(specs).collect()
# The largest gap between their predictions, then rls's fit.
gap = (out["rls"].struct.field("pred_y") - out["sum"].struct.field("pred_y")).abs().max()
print(f"the two predictions differ by less than 1e-9 on every row: {gap < 1e-9}")
print("planted at the last row: [0.50, 2.00, -1.00]")
print("rls's coefficients at the last row:", [round(c, 2) for c in out["rls"].struct.field("coef").drop_nulls()[-1]])
```

```text
the two predictions differ by less than 1e-9 on every row: True
planted at the last row: [0.50, 2.00, -1.00]
rls's coefficients at the last row: [0.52, 1.92, -0.99]
```

#### See also

When `rls`'s fit is not the one you need:

- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), the same
  fit under `ridge_scale="sum"`, solved on a schedule, whose default
  `ridge` does not wind up. The docstring's Notes, under *Wind-up*, measure
  how a feature that holds one value winds `rls` up.
- [`huber`](#huber--robust-regression), with
  `ridge_scale="sum"`, this fit with each row down-weighted by its
  residual.
- [`kalman`](#kalman--random-walk-β-dynamic-linear-model), for
  coefficients that drift as a random walk.
- [Fit targets that go missing on different
  rows](how-to/fit-targets-that-go-missing-on-different-rows.md), for
  targets null on different rows, which share one factor here and so are
  learned only where all are present.

#### Notes

**`rls` matches `ewridge(ridge_scale="sum")` solved on every row, to better
than 1e-9,** as the Example shows. At unit weights on a row clock it is
padasip's `FilterRLS`, which keeps the textbook inverse `P`, and agrees
with it to 1.6e-14 relative. On NIST's `Filip`, a degree-10 polynomial
whose correlation matrix is conditioned at `1.5e16`, `rls` keeps 7 digits,
where `ewridge` withholds the fit. It folds each row into a triangular
factor by rotations, so it never forms `zᵀz` and keeps the digits that
squaring the design loses.

### `lasso` — lasso path, its penalty chosen as it runs

*API:* [`rheon.spec.lasso`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.lasso) — *Rust:* [`lasso.rs`](../crates/rheon-core/src/lasso.rs) — *Outputs:* [fields](OUTPUTS.md#lasso)

A lasso or elastic-net path on running sums, with the penalty chosen as
the stream runs.

Use `lasso` for a sparse fit, where some features should get a coefficient
of exactly zero. Every penalty of the path is predicted, and
`penalty_selected_<t>` names the one with the lowest exponentially weighted
out-of-sample squared error so far, at no extra solve. Use
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) for a dense
fit, and [`ftrl`](#ftrl--online-logistic-regression) for a sparse logistic
fit of a 0/1 target.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.lasso),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#lasso). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes).

#### The update

The running sums are `ewridge`'s, centered. On the solve schedule, for each
penalty `l` in `penalties`, the model runs coordinate descent on the
standardized sums. The descent starts the first penalty from the last
solve's fit, and each penalty after it in `penalties` from the one before it. `C` is the
features' correlation matrix, `s_i` feature `i`'s standard deviation,
`c_i = cov(x_i, y)/s_i` its covariance with the target over its standard
deviation, and `b_i = β_i·s_i` its coefficient in standardized units. `s_y`
is the target's spread: its standard deviation over the rows the fit
reads, or its root mean square without an intercept. A sweep sets, for
every `i`:

```
ρ_i = c_i − Σ_{j≠i} C_ij b_j
b_i = soft(ρ_i, l·l1_ratio) / (C_ii + l·(1 − l1_ratio))      soft(v, t) = sign(v)·max(|v| − t, 0)
stop after a sweep that moves no b_i by tol·max(s_y, maxᵢ |c_i|) or more, or after max_iter sweeps
β_i = b_i / s_i         β_0 = ȳ − m·β         m the features' means, ȳ the target's
```

Without an intercept, the sums are raw moments, each feature is scaled by
its root mean square, and no intercept is recovered.

#### Example

**The penalty that has predicted best so far is reported on every row.**
Here 3 of 20 features carry the target, and the penalty in force at the
last row keeps exactly those three:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# 20 features, of which only x0, x1 and x2 carry weight in the target.
rng = np.random.default_rng(3)
n, k = 5_000, 20
x = rng.standard_normal((n, k))
y = 2.0 * x[:, 0] - 1.5 * x[:, 1] + 1.0 * x[:, 2] + rng.standard_normal(n)
names = [f"x{i}" for i in range(k)]
pl.DataFrame(x, schema=names).with_columns(y=y).write_parquet("wide.parquet")

# A path of penalties, the best so far chosen as the stream runs.
las = rp.spec.lasso(
    "las", targets=["y"], features=names,
    half_life=float("inf"),                  # every row counts the same
    penalties=[0.3, 0.1, 0.03, 0.01],        # the penalties, decreasing
)
bank = rp.ModelBank([las])
out = pl.concat(bank.fit_predict_batches(pl.scan_parquet("wide.parquet")))   # the output, chunk by chunk
# The penalty chosen at the last row, and the features it keeps.
chosen = out["las"].struct.field("penalty_selected_y")[-1]
slopes = bank.coef().filter(pl.col("penalty") == chosen, pl.col("term") != "intercept")
kept = slopes.filter(pl.col("coef") != 0.0)
print(f"penalty in force at the last row: {chosen}")
print("planted: x0 2.00, x1 -1.50, x2 1.00, the other 17 0")
print(f"coefficients left nonzero: {kept.height} of {slopes.height}:",
      {r["term"]: round(r["coef"], 2) for r in kept.iter_rows(named=True)})
```

```text
penalty in force at the last row: 0.03
planted: x0 2.00, x1 -1.50, x2 1.00, the other 17 0
coefficients left nonzero: 3 of 20: {'x0': 1.97, 'x1': -1.44, 'x2': 0.99}
```

#### See also

When `lasso`'s fit is not the one you need, or to run its descent outside a
bank:

- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), the
  running sums this model solves, under a ridge.
- [GRAM.md](GRAM.md), whose `rheon.gram.lasso_path` runs the same descent
  offline on a Gram.
- [`ftrl`](#ftrl--online-logistic-regression), a sparse fit of a 0/1
  target.
- [Forget rows past a window](how-to/forget-rows-past-a-window.md), which cuts the sums and the
  selection's errors at the same row, so a feature constant inside the
  hard window gets a coefficient of exactly zero.
- [Choose columns on the lasso path](how-to/choose-columns-on-the-lasso-path.md),
  to scale `penalties` with the target, rank them on every error so far,
  and give a descent more sweeps.

#### Notes

**The selection scores every prediction, the ones `min_weight` withholds
included.** If the gate held the selection back as well, the penalty
chosen would move with `min_weight`. With six penalties on a stream with
t(4) noise, such a selection differs after 3,000 rows from the choice at
`min_weight=0` on 6 of 50 seeds at the default `min_weight`, and on 13 at
a `min_weight` of 50.

**With no decay, a point of `lasso`'s path is statsmodels' elastic net on
the same rows.** At a penalty of 0.1, with a third feature that is noise,
the coefficients agree within 1e-7 and both zero the noise feature. At a
penalty of 0 the predictions are least squares from numpy, within 1e-6.
Its solves are also held to the lasso's optimality conditions
(Karush-Kuhn-Tucker), and `rheon.gram.lasso_path` to scikit-learn's
`enet_path`.

### `kalman` — random-walk-β dynamic linear model

*API:* [`rheon.spec.kalman`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.kalman) — *Rust:* [`kalman.rs`](../crates/rheon-core/src/kalman.rs) — *Outputs:* [fields](OUTPUTS.md#kalman)

A regression whose coefficients drift, tracked by a Kalman filter.

Use `kalman` when a coefficient moves at a drift rate of its own, faster
than one half-life for the whole fit can follow, or when a coefficient no
row supports should be forgotten (`revert_half_life`). Use
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) for
coefficients that hold still over a half-life, and
[`rls`](#rls--recursive-least-squares) for an exact fit whose prior fades.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.kalman),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#kalman). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes).
Here `half_life` decays the noise estimate and the features'
standardization, and `coef_half_life` sets how fast a coefficient may
drift.


**Forecasts at horizons.** `forecast_spans` asks for the prediction a row of
this row's features would get a span `Δ` of the clock past the row, with no
row between, as the filter would make it. `forecast_<h>_<t>` is that
prediction and `forecast_sum_<h>_<t>` its integral over the span.
`forecast_var_<h>_<t>` is its variance, from the covariance carried through
the transition over `Δ` and the drift of the clock since the target's last
observation plus `Δ`. At `Δ = 0` the
forecast is `pred_<t>`. The features are held at the row's: the forecast is
the regression's at them. Under the default random walk the forecast does
not move with `Δ` and only its variance grows; under `revert_half_life` it
reverts too. Against `filterpy`'s filter run on the same rows, unstandardized
with a given noise, the three agree to 1.4e-15.

#### The update

Per target `j`, `β_j` are the coefficients and `R_j·P_j` their covariance.
`P_j` is held in units of the observation noise `R_j`, the target's
exponentially weighted residual variance, or `obs_var` where given. `z̃`
is `z` with each feature standardized against the running means `m` and
standard deviations `s` of the rows before it, `[1, (x − m)/s]` (`x/r`
without an intercept, with `r` each feature's root mean square), and `z`
itself under `standardize=False`. `d` is the row's clock step, capped
at `gap_cap`, and `κ_i` is coefficient `i`'s `revert_half_life`. `Q` is the
process noise in the same units as `P_j`: `(ln 2/h_i)²` from a
`coef_half_life` of `h_i`, or `q_i/R_j` from `q`:

```
β_j ← Φβ_j     P_j ← ΦP_jΦ        Φ = diag(2^(−d/κ_i))        the reversion, every row
D_j ← D_j + d                                                the clock since y_j was last observed
P_j ← P_j + Q·D_j²,  D_j ← 0                                 the drift, on a row that observes y_j
v   = z̃ᵀP_j z̃ + 1/w                 the innovation's variance
K   = P_j z̃ / v                     the gain
β_j ← β_j + K(y_j − z̃ᵀβ_j)          P_j ← P_j − K z̃ᵀP_j
```

A row's weight scales its observation's precision: the observation's
variance is `R_j / w`. A reverting coefficient's drift over a gap `D` is
`Q_i·g_i(D)²` in place of `Q_i·D²`, with `g_i(D) = (1 − 2^(−D/κ_i))/θ_i` and
`θ_i = ln 2/κ_i`. It is `Q_i·D²` for a short gap, and never more than
`Q_i/θ_i²`, so a reverting coefficient's uncertainty stays bounded across a
run of null targets. The docstring's update gives the form under
`obs_var`, which runs on the covariance itself.

The last line subtracts two numbers of `P_j`'s size to leave one of the
posterior's, so it rounds at `u·p0`, with `u` the unit roundoff: at
`p0 = 1e12` it loses digits to the fourth place. `covariance_form =
"square_root"` holds `P_j` as an upper-triangular factor, `P_j = C Cᵀ`,
and updates `C` column by column, with `f = Cᵀz̃`, `a_0 = 1/w` and
`a_j = a_{j−1} + f_j²` (Carlson's update):

```
C_j ← √(a_{j−1}/a_j)·C_j − f_j/√(a_{j−1}a_j) · Σ_{i<j} C_i f_i
```

`C`'s entries are of size `√p0`, so it rounds at `u·√p0`. Use it for a
vague prior, `p0` of about `1e8` or more. On arm64 it takes 1.14 to 1.72
times the covariance form's time per row with no drift, and 2.4 to 16
times under `coef_half_life` or `q` at 5 to 50 coefficients, since the
drift then updates `C` once per coefficient. At `p0 = 1e15`, against
statsmodels' Kalman filter under its exact diffuse initialization, its
predictions agree to 3.3e-13 where the covariance form's are up to 0.78
off.

#### The fixed-lag smoother

`smooth_lag = L` adds, on the row that completes `L` observations of a
target after an earlier one, the coefficients at that earlier observation
estimated from every row up to this one. They are `smoothed_coef`, laid
out like `coef`, with `smoothed_se_coef` under `emit_smoothed_se`, and
`smoothed_for_<t>` is the clock of the row each target's value describes.
`coef` can only estimate a row's coefficients from the rows up to it; the
smoothed value also reads the `L` observations after it. An observation is
a row on which the target is present at a weight above 0, so a row of
weight 0 or a null target counts in no lag.

After observation `t`'s update the Rauch-Tung-Striebel recursion runs back
`L` observations. `β_n`, `P_n` is the filtered pair after observation `n`,
`β⁻`, `P⁻` the prior observation `n + 1`'s update starts from, and `F` the
map the filter applies to the coefficients between the two: each row's
`Φ`, and each change of the standardizer's coordinates.

```
G_n = P_n Fᵀ (P⁻_{n+1})⁻¹
m ← β_n + G_n (m − β⁻_{n+1})          from m = β_t
V ← P_n + G_n (V − P⁻_{n+1}) G_nᵀ     from V = P_t
```

A step reads only observation `n`'s filtered pair and the smoothed pair at
`n + 1`, so `L` steps from `t` are exactly the smoother of every row up to
`t`, read at `t − L`. Each observation keeps its gain, computed once from
the next observation's prior, and the pass runs on the rows that emit.
The state is `L` gains of `k²` numbers a covariance, `k` the coefficients.
The settings bound it, and it is held to `max_state_bytes` (256 KB a group
by default) before a row is read.

The smoothed value is never fed back: every prediction is the filter's, to
the bit. It is written on the line of the row that completed the lag, named
for the row it describes, and never on that row's own line, where a reader
would take it for something known then. Used as a target for the described
row, it needs an `embargo` on the downstream spec of at least the lag's
span. `smooth_lag` is refused beside this spec's own `embargo` or
`label_clock`, which release several held rows on one row, and under
`covariance_form = "square_root"`. Where the filter does not continue
itself (a covariance sized again, a change of coordinates refused, a prior
whose factorization fails) the recursion does not cross, and the
observations before are smoothed with the rows up to the break. The last
`L` observations, whose lags no row has completed, are read from the state
with [`bank.smoothed_tail()`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.smoothed_tail).

#### Example

**A coefficient that may drift follows a jump that running sums only
average in.** Here a slope jumps from 1 to 3 at row 2,000. `kalman` lets
half a coefficient's memory go in 20 clock units, and `ewridge`'s sums
forget at the spec's half-life of 200:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# A slope on x that jumps from 1 to 3 at row 2,000, one row a clock unit.
rng = np.random.default_rng(4)
n = 3_000
x = rng.standard_normal(n)
slope = np.where(np.arange(n) < 2_000, 1.0, 3.0)
y = slope * x + 0.5 * rng.standard_normal(n)
pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y}).write_parquet("jump.parquet")

# kalman, whose slope may drift, beside ewridge's running sums.
common = dict(targets=["y"], features=["x"], clock="t", gap_cap=10.0, half_life=200.0, coef_every=0)
specs = [
    rp.spec.kalman("kalman", coef_half_life=20.0, **common),            # a coefficient may drift
    rp.spec.ewridge("ewridge", max_rows_between_solves=1, **common),    # solved on every row
]
out = pl.scan_parquet("jump.parquet").rheon.fit_predict(specs).collect()
# Each model's slope just before the jump and after it.
for row in (1_999, 2_010, 2_050, 2_200):
    k = out["kalman"].struct.field("coef")[row][1]     # [intercept, slope] after the row
    e = out["ewridge"].struct.field("coef")[row][1]
    print(f"row {row}: truth {slope[row]:.0f}, kalman {k:.2f}, ewridge {e:.2f}")
```

```text
row 1999: truth 1, kalman 1.00, ewridge 1.01
row 2010: truth 3, kalman 1.56, ewridge 1.09
row 2050: truth 3, kalman 2.76, ewridge 1.45
row 2200: truth 3, kalman 3.04, ewridge 2.05
```

Ten rows after the jump `kalman` has covered over a quarter of it, and 200 rows
after, all of it. `ewridge` still averages the slope over its half-life of
200 rows.

#### See also

When `kalman`'s fit is not the one you need, or to use it past the data:

- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), for
  coefficients that hold still over a half-life.
- [`rls`](#rls--recursive-least-squares), the fit this filter gives with
  no drift, a fixed noise, `standardize=False`, no `revert_half_life` and
  `half_life=inf`, at `delta = 1/p0`.
- [`bank.predict()`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.predict),
  which moves the coefficients by `Φ` over the clock since the last
  learned row.
- [Choose a Kalman filter's drift speed](how-to/choose-a-kalman-drift-speed.md),
  to set each coefficient's drift and reversion, and how far a prediction
  past the data may move.
- [Scale a feature by its spread](how-to/scale-a-feature-by-its-spread.md),
  for a coefficient per current standard deviation, under
  `standardize=False`.

#### Notes

**Under `coef_half_life`, the gain never reads the noise estimate.** A
filter whose gain read it would weigh each row by the inverse of the noise
as it stood before the row, so a burst of large residuals would set the
weights. The docstring's Notes, under *The noise*, give such a run with no drift: a
burst in a fifth of the first 500 rows held the slopes at `[-25.5, 33.6]`,
where least squares gives `[4.1, -1.4]`. Weighing rows by a running noise
estimate is also volatility weighting. On the intraday returns of
[VALIDATION.md](VALIDATION.md), section 5, the filter with the estimate in
its gain reached R² −0.059 on the next minute's return, against −0.070 for
this one, and −0.075 against −0.101 on five minutes'. So on returns whose
volatility clusters, keeping the estimate out of the gain gives up some
R², in exchange for a fit that weighs a burst of large residuals as it weighs
any other rows.

**Without `obs_var`, the intercept's prior is diffuse, so a level in the
target moves every prediction by that level from the first row.** The
first row that observes the target puts its whole innovation into the
intercept. That is the limit of an intercept prior whose variance grows
without bound (Durbin and Koopman, *Time Series Analysis by State Space
Methods*, §5.2). A filter whose prior was sized by the noise estimate took
the level in through its first innovations. At a level of `1e4`, with a
`coef_half_life` of 200 and a half-life of 300, its predictions stood 0.019 to 0.032
of the target's spread off the unshifted fit over rows 3,000 to 6,000.
Under `coef_half_life` and without `obs_var`, neither the gain nor the
prior reads the noise, so the warm-up does not depend on the target's
units.

**With `standardize=False`, `q` all zeros, no `revert_half_life` and a
fixed `obs_var=`, the filter is exactly Bayesian linear regression,** the ridge with penalty
`1/p0`: river's `BayesianLinearRegression`, without an intercept, agrees
within 1e-12. Standardized, each prior waits for its feature's scale, so it
is not that ridge. Held to filterpy's `KalmanFilter`, configured to
match, an explicit `q`, a fixed `obs_var` with `p0`, and a
`coef_half_life` per coefficient agree to 1.1e-15 relative, and no
intercept under `standardize` to 1.0e-14. `share_p=True`
changes no prediction where every target is on every row, apart from under
`q`: on section 4 of
[VALIDATION.md](VALIDATION.md) it and a `P` per target score the same, to
the bit.

### `huber` — robust regression

*API:* [`rheon.spec.huber`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.huber) — *Rust:* [`robust.rs`](../crates/rheon-core/src/robust.rs) — *Outputs:* [fields](OUTPUTS.md#huber)

Huber regression on running sums, which down-weights the rows a prediction
misses by far.

Use `huber` in place of
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) where a
share of the rows are outliers that would pull a least-squares fit. Use
[`quantile`](#quantile--robust-regression) to fit a quantile of the
target, such as the median, and [`sgd`](#sgd--stochastic-gradient-descent)
with `loss="huber"` where learning a row must take O(k) time for `k`
features: this model takes O(k²) time.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.huber),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#huber). It takes
`ewridge`'s `ridge=` as a single number, and refuses a list. The
parameters every spec takes are in [What every spec
takes](#what-every-spec-takes).

#### The update

`e` is the row's residual against the fit before the row is learned, so
the weight the row is learned at stays out-of-sample. `σ` is the target's
residual scale as the row arrives: by default the MAD (median absolute
deviation), the exponentially weighted median of `|e|` over
`Φ⁻¹(3/4) = 0.6745`. `δ` is `huber_delta`. The row enters [`ewridge`'s](#ewridge--exponentially-weighted-ridge-regression)
update equations for `S_j`, `η_j` and `W_j` at its weight cut
by its residual, and the coefficients `β_j` are solved on `ewridge`'s
schedule:

```
w_r = w · min(1, δσ / |e|)           the row's Huber weight; w itself until σ is above 0
S_j, η_j, W_j:  ewridge's update equations, at weight w_r in place of w
solve, ridge_scale="mean":  (S_j + ridge·D) β_j = η_j          D = I with a 0 at the intercept
solve, ridge_scale="sum":   (W_j·S_j + p·ridge·I) β_j = W_j·η_j
                            W_j after the Huber weights, and the intercept penalized too
```

`z`, `w`, `S_j` and `η_j` are as in the [notation table](#models). `W_j` is
the weight behind `S_j` and `η_j`, as in `ewridge`'s update, and `p` the
product of every decay the stream has applied.

The weights are per target, so `huber` keeps one set of sums per target,
where `ewridge` shares one set across targets. `weight_sum`, and each
target's weight that `min_weight` reads, count the raw weights `w`. When a
target's count of learned rows reaches `huber_refit_rows`, its first rows
are refit once, to convergence, and the stream goes on online from that
fit. The docstring's Notes give the refit, under *The refit of the first
rows*, and the scale `σ`, under *The scale*.

#### Example

**A few wild targets pull a least-squares slope. The default Huber fit and
the median hold near the truth, and a Huber fit drawn in a root mean square
does not.** Here about 2% of the rows are 50 too high, all where `x` is
above 0, so they tilt the line:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# y = 1 + 2x plus noise, and about 2% of the rows wild: 50 too high, on rows where x is above 0.
rng = np.random.default_rng(5)
n = 10_000
x = rng.standard_normal(n)
y = 1.0 + 2.0 * x + 0.5 * rng.standard_normal(n)
wild = (rng.random(n) < 0.04) & (x > 0)
y[wild] += 50.0
pl.DataFrame({"x": x, "y": y}).write_parquet("wild.parquet")

# Least squares beside the two robust models, every row weighed the same.
common = dict(targets=["y"], features=["x"], half_life=float("inf"))
specs = [
    rp.spec.ewridge("ewridge", **common),
    rp.spec.huber("huber", **common),                         # huber_delta=1.345 and huber_scale="mad", the defaults
    rp.spec.huber("huber_rms", huber_scale="rms", **common),  # the cut drawn in the residuals' root mean square
    rp.spec.quantile("median", quantile=0.5, **common),
]
bank = rp.ModelBank(specs)
bank.fit(pl.scan_parquet("wild.parquet"))
# Each model's slope, beside the truth.
print(f"wild rows: {wild.sum()} of {n}; the true slope: 2")
for r in bank.coef().filter(pl.col("term") == "x").iter_rows(named=True):
    print(f"{r['spec']}: slope {r['coef']:.2f}")
```

```text
wild rows: 218 of 10000; the true slope: 2
ewridge: slope 2.89
huber: slope 2.02
huber_rms: slope 2.18
median: slope 2.03
```

`huber` still learns from a wild row, at a weight that shrinks with its
miss, so it lands near the truth and not exactly on it. The median takes
only the sign of a row's miss beyond its band (see
[`quantile`](#quantile--robust-regression)), so a wild row counts no more
than an ordinary one. `huber_rms` draws its cut in a scale that the wild
rows themselves widen, so it lets more of their pull through: a root mean
square is not a robust scale.

#### See also

When `huber`'s fit is not the one you need, or to see its loss from
another side:

- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), the
  least-squares fit on sums of the same form.
- [`quantile`](#quantile--robust-regression), a fit of a quantile of the
  target, on the same sums, scale and refit.
- [`sgd`](#sgd--stochastic-gradient-descent) with `loss="huber"`, the same
  loss with a step per row in O(k) time.
- [`rls`](#rls--recursive-least-squares), whose predictions `huber` gives within 1e-9
  with `ridge_scale="sum"`, `huber_delta=inf` and `solve_every=0`,
  where `rls` has `delta=ridge`.
- [Fit a Huber regression on a Gram](how-to/fit-huber-on-a-gram.md), a
  Huber fit from one shared Gram, two queries over the rows per pass.
- [Bound a run's memory](how-to/bound-memory.md#bound-the-rows-a-refit-holds),
  to bound the refit buffers of all a spec's groups with `refit_budget`.
- [Reproduce an earlier version's fit](how-to/reproduce-an-earlier-versions-fit.md),
  to come close to the numbers of a version before the refit, with
  `huber_refit_rows=0`.

#### Notes

[The robust scale and the refit of the first
rows](explanation/robust-scale.md) explains Huber's loss and why its
scale must be robust. It measures the streams one pass leaves captured,
the refit over 40 seeds, and what a robust scale gives up when the level
shifts.

**A refit runs once, and a row after it takes the time it did, within
2%.** The default count grows with the features, 10 rows a feature past
50, so that the refit has rows enough for its coefficients. It stops at
100 features, where a target's rows would pass 0.8 MB. A count given is
taken at any width. A count given below the number of coefficients, such
as 200 rows at 1,000 features, leaves the fit the ridge's, and the
iteration runs to its cap of 200 passes: about 9 s once, which
`solve_failures` counts. A cap of 50 passes would take about 2 s there,
but 200 is where the slow refits of heavy one-sided outliers end. Over
3,000 streams of 50 to 500 rows, with up to 45% of the rows on one side
moved by a level, 50 passes ran out on 127 streams for `huber` and 200 on
8, the slowest refit taking 428 passes. For `sgd(loss="huber")` they ran
out on 149 and 9, all nine past 39%.

**`huber`'s one pass is held to a numpy reference of its own recursion
within 1e-12 of `1 + |pred|` (measured at most 1.9e-15), and its refit to statsmodels' `RLM` on the same rows within
1e-5.** At `huber_delta=inf` with `ridge_scale="sum"`, solved on every row,
it gives `rls`'s predictions with `delta=ridge` within 1e-9, as
`ewridge(ridge_scale="sum")` does. Against a batch fit of the same
objective over a whole stream, scikit-learn's `HuberRegressor`, it lands
close but not equal, for two reasons:

| reason | applies to |
|---|---|
| each row's weight was set by the fit before it | each row past the refit, and every row where the refit is off or never runs |
| `σ` is an exponentially weighted median, read exactly from the held residuals while the refit waits, and from a sketch after it or where the refit is off, where a batch fit takes the median of the final residuals | every row (under `"rms"`, `σ` is a root mean square) |

### `quantile` — robust regression

*API:* [`rheon.spec.quantile`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.quantile) — *Rust:* [`robust.rs`](../crates/rheon-core/src/robust.rs) — *Outputs:* [fields](OUTPUTS.md#quantile)

Quantile regression on running sums, the fit that a share `quantile` of the
targets fall below.

Use `quantile` for a conditional median, which outliers cannot pull, or
for a tail, such as the 0.9 quantile of a return, where a mean says
nothing. Use [`huber`](#huber--robust-regression) for a mean that outliers
should not pull, and [`sgd`](#sgd--stochastic-gradient-descent) with
`loss="quantile"` where learning a row must take O(k) time for `k`
features: this model takes O(k²) time.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.quantile),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#quantile). It takes
`ewridge`'s `ridge=` as a single number, and refuses a list. It takes no
`feature_sets`, since its rows outside the band step the fit through one
system per target. The parameters every spec takes are in [What every spec
takes](#what-every-spec-takes).

#### The update

`quantile` takes one Newton step per row on the check loss, smoothed by a
uniform kernel of half-width `h`. The step is linearized at the fit the row
was predicted with, so it stays out-of-sample. `e` is the row's residual,
`τ` is `quantile`, and `σ` is the target's residual scale, the MAD: the
exponentially weighted median of `|e|` over `Φ⁻¹(3/4) = 0.6745`. `k` is the
number of coefficients, and `n` the target's present rows, counted one
each and decayed, whatever their weights. `A` is the band's system the
solve factorizes, and `u` the row's deviation from the band's means.
`z`, `w`, `S_j` and `η_j` are as in the [notation table](#models):

```
ψ_τ(e) = τ − 1{e < 0}
h      = σ · max(quantile_eps, (k/n)^(2/5))      the band's half-width
|e| <  h:  a least-squares row (ewridge's update equations for S_j and η_j), with target  y + 2h(τ − ½)
|e| ≥  h:  nothing to S_j;  w · 2h · ψ_τ(e) · z  into η_j, over the band's weight,
           capped at  |e| / (1 + uᵀA⁻¹u)
```

The cap reads the row's full leverage against `A`, so a step moves the
fit at the row by at most the row's residual. Under three rows per
coefficient, and until a target has a `σ` above 0, a row is a
least-squares row. Like `huber`, `quantile` keeps one set of sums per
target, and refits each target's first rows once, to convergence, when its
count of learned rows reaches `quantile_refit_rows`. The docstring's
Notes give the warm-up under *Warm-up and rebuilding*, the band's floor
under *The band*, and the refit under *The refit of the first rows*.

#### Example

**`quantile=0.9` fits the line that 90% of the targets fall below.** Here the noise
is normal with a spread of 1, so the 0.9 quantile of `y` is the line
`1 + 1.28 + 2x`:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# A line with normal noise of spread 1: its 0.9 quantile is 2.28 + 2x.
rng = np.random.default_rng(4)
x = rng.standard_normal(20_000)
y = 1.0 + 2.0 * x + rng.standard_normal(20_000)
pl.DataFrame({"x": x, "y": y}).write_parquet("tail.parquet")

q90 = rp.spec.quantile(
    "q90", targets=["y"], features=["x"], half_life=float("inf"),
    quantile=0.9,                                  # the level to fit
)
bank = rp.ModelBank([q90])
bank.fit(pl.scan_parquet("tail.parquet"))          # learn every row, keep no output
# The line found, beside the true one.
print("the true line: intercept 2.28, slope 2.00")
c = {r["term"]: r["coef"] for r in bank.coef().iter_rows(named=True)}
print(f"found: intercept {c['intercept']:.2f}, slope {c['x']:.2f}")
```

```text
the true line: intercept 2.28, slope 2.00
found: intercept 2.31, slope 1.99
```

The median takes only the sign of a row's miss beyond its band, so a wild
row counts no more than an ordinary one. [`huber`'s
entry](#huber--robust-regression) runs `quantile(quantile=0.5)` beside
`huber` and least squares: the median's slope is 2.03 against a true 2,
where least squares gives 2.89.

#### See also

When `quantile`'s fit is not the one you need:

- [`huber`](#huber--robust-regression), the robust fit of the mean, whose
  scale sketch this model reads.
- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), the
  least-squares fit on sums of the same form.
- [`sgd`](#sgd--stochastic-gradient-descent) with `loss="quantile"`, a
  step per row in O(k) time.

#### Notes

[The refit of the first
rows](explanation/robust-scale.md#the-refit-of-the-first-rows) measures
`quantile`'s refit against statsmodels' `QuantReg` over 50 streams. The
refit's buffer, its count's default and `refit_budget` are
[`huber`'s](#huber--robust-regression).

**`quantile`'s one pass is held to a numpy reference of its own recursion
within 1e-9 of `1 + |pred|`.** Against a batch fit of the same objective over a whole
stream, statsmodels' `QuantReg`, it lands close but not equal, for three
reasons:

| reason | applies to |
|---|---|
| each row was taken by the fit before it | each row past the refit, and every row where the refit is off or never runs |
| `σ` is an exponentially weighted median, read exactly from the held residuals while the refit waits, and from a sketch after it or where the refit is off, where a batch fit takes the median of the final residuals | every row |
| the check loss is smoothed over the band | every row |

The docstring's Notes, under *Why a Newton step*, measure the step against
`QuantReg` at the median of a skewed noise, and under *The scale*, the MAD
against a root mean square.

### `expectile` — expectile regression

*API:* [`rheon.spec.expectile`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.expectile) — *Rust:* [`expectile.rs`](../crates/rheon-core/src/expectile.rs) — *Outputs:* [fields](OUTPUTS.md#expectile)

Expectile regression, fitted one row at a time by reweighted least
squares: the squared error with the rows above the fit weighed `tau` and
those at or below it `1 - tau`.

Use `expectile` for a tail that should weigh how far the rows past it
lie, not only how many there are: the 0.9 expectile of a loss, a latency or
a demand. `expectile=0.5` is the least-squares fit. Use
[`quantile`](#quantile--robust-regression) for the level a share of the
rows fall below, which no single row can move far.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.expectile),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#expectile). One spec
fits one level: each level weighs every row by its own sides, so two levels
share no sums, and two specs cost what one spec of two levels would. The
parameters every spec takes are in [What every spec
takes](#what-every-spec-takes).

#### The update

`a(r)` is `tau` for a residual above 0 and `1 - tau` at or below it, and
`z`, `w` are as in the [notation table](#models). Each target keeps a
mean-form Gram `G`, a cross-moment `c` and the asymmetric weight `S`; `λ`
is the decay over the clock since the last row that counted:

```
pred = zᵀβ                    r = y − pred       (before the update)
a    = a(r), or ½ before the target's first fit
S    = λ S + w a
W    = λ W + w
G    = G + (w a / S)(z zᵀ − G)
c    = c + (w a / S)(z y − c)
β    = (G + ridge (W / S) D)⁻¹ c
```

`D` is the slopes' diagonal, so `ridge` penalizes the slopes by the
target's weight `W`, the intercept spared. Its default, 1e-6, keeps
every solve defined from the first row; at 0 the fit is the exact
loss's and waits for a row per coefficient.

A row's own update cannot move its residual across zero, so one solve a
row is the whole step. What one pass cannot do is judge an earlier row
again once the fit has crossed it. Each target's first `refit_rows` rows
are therefore fitted exactly once, and the sums rebuilt at the sides the
exact fit gives them. `refit_span` is the same refit over a span of the
clock from the target's first learned row, the row that reaches the span its
last. Its rows are checked against `max_state_bytes` as they come. A
window, `window_rows` rows or `window_size` of the
clock, instead refits the rows it holds on every row. The exact fit
repeats a weighted least-squares solve at the current sides, judging every
side again, until no side changes, halving a step that does not lower the
loss.

#### Example

**`expectile=0.9` fits the line the large rows pull up.** Here the noise is
an exponential of mean 1, whose 0.9 expectile is about 2.03, so the line is
`1 + 2.03 + 0.5x`:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# A line with skewed noise: its 0.9 expectile is about 3.03 + 0.5x.
rng = np.random.default_rng(7)
x = rng.standard_normal(20_000)
y = 1.0 + 0.5 * x + rng.exponential(1.0, 20_000)
pl.DataFrame({"x": x, "y": y}).write_parquet("skewed.parquet")

# The 0.9 expectile, learned from every row.
upper = rp.spec.expectile(
    "upper", targets=["y"], features=["x"], half_life=float("inf"),
    expectile=0.9,                                 # the level to fit
)
bank = rp.ModelBank([upper])
bank.fit(pl.scan_parquet("skewed.parquet"))        # learn every row, keep no output
# The line found.
c = {r["term"]: r["coef"] for r in bank.coef().iter_rows(named=True)}
print(f"found: intercept {c['intercept']:.2f}, slope {c['x']:.2f}")
```

```text
found: intercept 3.03, slope 0.51
```

#### See also

When `expectile`'s fit is not the one you need:

- [`quantile`](#quantile--robust-regression), the level a share of the rows
  fall below.
- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), the
  least-squares fit, which `expectile=0.5` is, with its diagnostics.
- [`huber`](#huber--robust-regression), a mean that outliers cannot pull.

#### Notes

**The fit is statsmodels' `RLM` with `MQuantileNorm(tau,
LeastSquares())`.** The refit of the first rows and every row of a window
match it on the same rows, to 2.3e-14 and 7.6e-15 relative. A window of the
constant alone matches `scipy.stats.expectile` within 2 ulps of the largest
target. At `tau = 0.5` the one pass is the decayed least-squares
fit to rounding. Elsewhere the one pass is a statistical approximation.
Measured over 10 seeds with decay, its fitted values stood 0.02 to 0.4 of
the batch fit's own sampling error from the exact decayed fit on average,
and 1.2 of it at worst (500 rows, `tau = 0.9`). Without
decay, the refit of the first rows cut that gap fivefold at 500 rows.

**The state is bounded by the spec.** The sums take `k(k+1)/2 + 2k + 4`
numbers a target for `k` coefficients. A refit buffer takes `refit_rows ·
(features + 2)` numbers a target until it runs, and `window_rows` rows of
`8 (features + targets + 1) + targets` bytes each. Both are checked against
`max_state_bytes`, 256 KB by default and at most 64 MiB, when the spec is
built. A `window_size` window holds the rows its span has, which the data
decides, so it is checked as the rows go in: a warning at half of the cap,
once a group, and a refusal at it, each naming the group and the bytes.

**A row of weight 0 counts in no clock.** It is predicted as any row is,
and the next row decays as if it were not there, its gap taken once from
the last row that counted. It is in no window, and in no `weight_sum`.

### `sgd` — stochastic gradient descent

*API:* [`rheon.spec.sgd`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.sgd) — *Rust:* [`sgd.rs`](../crates/rheon-core/src/sgd.rs) — *Outputs:* [fields](OUTPUTS.md#sgd)

Stochastic gradient descent under a choice of losses, with one gradient
step a row.

Use `sgd` when learning a row must take only O(k) time for `k` features:
it takes one gradient step a row and never solves. It fits count targets
under `loss="poisson"` in O(k) time a row, where
[`glm`](#glm--generalized-linear-models-poisson-gamma-binomial-tweedie)'s
`family="poisson"` takes a Newton step in O(k³), and its coefficients can
be bounded or summed to a constant, as portfolio weights are. Use
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) for an exact
fit in O(k²) time a row, [`huber`](#huber--robust-regression) or
[`quantile`](#quantile--robust-regression) for a robust or quantile fit
solved on running sums, and [`pa`](#pa--passive-aggressive-regression) for
steps with no learning rate to tune.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.sgd),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#sgd). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes).

#### The update

Here `p` is the prediction, `h` the inverse of the loss's link, `d` the
loss's gradient at `p`, `lrᵢ`
coefficient `i`'s rate on the row, and `Gᵢ` AdaGrad's running sum of its
squared gradients. With `standardize=False`, `z = [1, x]` and `β` is in the
caller's units, so a row takes this step:

```
eta = zᵀβ        p = h(eta)        d = dL/d eta
g₀ = d·w         gᵢ = d·zᵢ·w + l2·βᵢ     each clipped to ±clip_gradient·u; the intercept takes no l2
βᵢ -= lrᵢ·gᵢ     lrᵢ = learning_rate                           schedule="constant"
                     = learning_rate/(1 + λ·W)^power           schedule="inv_scaling"
                     = learning_rate/(√Gᵢ + 1e-8·u)            schedule="adagrad"
```

`λ·W` is the weight before the row, decayed to it, and `Gᵢ` decays by `λ`
too. So a row of weight 0 in between leaves the next row's rate where it
would be without that row.

`u` is the unit the gradient is in. Under `squared` and `huber`, whose
gradient is in the target's units, it is `s_y`, the target's own spread,
and there is no clip before the target has one. Under the other losses,
whose gradient is a sign, a count's residual or a probability's, it is 1.
So the clip and AdaGrad's epsilon mean the same at every scale of the
target.

The loss sets the link, whose inverse `h` gives the prediction, and the
gradient:

| loss | link | `p = h(eta)` | `dL/d eta` |
|---|---|---|---|
| `squared` | identity | `eta` | `p − y` |
| `huber` | identity | `eta` | `clamp(p − y, ±delta·s)` |
| `quantile` | identity | `eta` | `1{y < p} − τ` |
| `epsilon_insensitive` | identity | `eta` | 0 within `eps·s_y` of `y`, else `sign(p − y)` |
| `poisson` | log, for count targets | `exp(clamp(eta, ±30))` | `p − y` |
| `logistic` | logit, for 0/1 targets | `sigmoid(eta)` | `p − clamp(y, 0, 1)` |

`delta` is `huber_delta`, `τ` is `quantile` and `eps` is the tube's
half-width. `s` is the scale of the target's out-of-sample residuals, as
for `huber`: the exponentially weighted median of `|y − p|` over 0.6745
under the default `huber_scale="mad"`, or their root mean square under
`"rms"`. `s_y` is the target's own exponentially weighted standard
deviation.

Under `standardize`, the default, the step is taken on the standardized
row `z̃`, with `z̃ᵢ = (xᵢ − mᵢ)/sᵢ` and `mᵢ` and `sᵢ` column `i`'s running
mean and standard deviation. Without an intercept, `z̃ᵢ = xᵢ/sᵢ`, with `sᵢ`
its root mean square (not centered). The ridge reads `l2·sᵢ·βᵢ`. So
`gᵢ` is `d·z̃ᵢ·w + l2·sᵢ·βᵢ`. Once the scaler has warmed up, the step taken
on `z̃` is mapped into the caller's units:

```
βᵢ -= lrᵢ·gᵢ / sᵢ        β₀ -= lr₀·g₀ − Σᵢ mᵢ·lrᵢ·gᵢ / sᵢ
```

Under an identity link with an intercept, the intercept takes the running
mean's step in place of its gradient's. Here `W_y` is the target's learned
weight (undecayed, this row's in it), `a = w/W_y`, and `n` is Kish's
effective sample size of the target's rows:

```
squared, huber:                  β₀ += −a·d        while a is above the row's rate, or n ≤ 22
                                                   (under schedule="adagrad", only while n ≤ 22)
quantile, epsilon_insensitive:   β₀ += a·(y − p)   while n ≤ 22
```

The docstring's Notes, under *The target's level*, give the reason and
its measurements.

#### Example

**A count target takes the Poisson loss, whose log link makes `pred` a
rate.** Here the counts' log-rate is `0.5 + 0.3·x0 − 0.2·x1`, and adagrad
adapts each coefficient's rate to its own gradients:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# A count target: y is Poisson with log-rate 0.5 + 0.3·x0 − 0.2·x1.
rng = np.random.default_rng(6)
n = 50_000
x0, x1 = rng.standard_normal((2, n))
y = rng.poisson(np.exp(0.5 + 0.3 * x0 - 0.2 * x1)).astype(float)
pl.DataFrame({"x0": x0, "x1": x1, "y": y}).write_parquet("counts.parquet")

# The Poisson loss, fitted one gradient step per row.
counts = rp.spec.sgd(
    "counts", targets=["y"], features=["x0", "x1"], half_life=float("inf"),
    loss="poisson",                          # a chunk with a negative target is refused, naming the row
    schedule="adagrad", learning_rate=0.05,  # a rate per coefficient, adapted to its gradients
)
bank = rp.ModelBank([counts])
bank.fit(pl.scan_parquet("counts.parquet"))
print("truth:  {'intercept': 0.5, 'x0': 0.3, 'x1': -0.2}")
print("fitted:", {r["term"]: round(r["coef"], 2) for r in bank.coef().iter_rows(named=True)})
```

```text
truth:  {'intercept': 0.5, 'x0': 0.3, 'x1': -0.2}
fitted: {'intercept': 0.51, 'x0': 0.31, 'x1': -0.21}
```

Each coefficient the fit found is within about 0.01 of the log-rate's.

#### See also

When `sgd`'s steps are not the fit you need:

- [`pa`](#pa--passive-aggressive-regression), steps with no learning rate,
  with the same scaler and bounds.
- [`huber`](#huber--robust-regression) and
  [`quantile`](#quantile--robust-regression), the robust and quantile fits
  solved on running sums.
- [`ftrl`](#ftrl--online-logistic-regression), a logistic or linear fit
  whose L1 penalty zeroes coefficients.
- [Choose a learning rate](how-to/choose-a-learning-rate.md#choose-sgds-learning-rate),
  to set how many rows the coefficients remember.
- [Follow a level that moves](how-to/follow-a-level-that-moves.md#give-sgds-huber-scale-a-shorter-memory),
  to give `huber_scale_half_life` a shorter memory where the level can
  change.
- [Fit a count target](how-to/fit-a-count-target.md), under
  `loss="poisson"` and its `clip_gradient`.
- [Bound the coefficients](how-to/bound-the-coefficients.md#bound-the-slopes-of-sgd),
  to hold the slopes to `coef_min`, `coef_max` and `coef_sum`.

#### Notes

**The stability check reads a sample of rows, so it is built to resist one
wide row and to rest while the rate is calm.** The docstring's Notes,
under *A rate past its stability bound*, give the rule and the check.

**One wide row cannot hold the output.** Each gain after a target's first
enters the sums bounded, at most the larger of 2 and four times the
running mean `m` before it. A gain of 2 leaves its row's residual where it
was and adds nothing to the margin `Σ(q² − 2q)`. So while `m` is below 1/2,
no single row moves the verdict toward unstable. Above it, a row moves the
verdict by at most `16m² − 8m`. Entered whole, one row at 100 times a
feature's spread held a stable rate's output for 1,075 rows (raw features,
a rate of 0.01), and one at 10,000 times for 3,736.

**The check's memory counts rows, a half-life of 100 stepped rows, because
its verdict estimates an expectation from a sample of the rate's rows.** The error of a ratio of two means of `N` gains is about `CV(q)/√N`, where
`CV(q)` is the gains' standard deviation over their mean. That spread is
set by the spread of `|z|²`, which is the rows' and not the rate's or the
decay's.
The fit's own memory, about `k / mean(q)` rows for a coefficient, is a
handful of rows near the bound, too few to estimate from. As the check's
memory it withheld 6-11% of a stable rate's rows (at 0.77 times the bound),
and served a third to a half of a diverging one's (at 1.15 times). The
model's decay ages the target's weight and AdaGrad's sums, not the
coefficients, so it is no memory of the rate either.

**A calm check rests.** Under the squared and Huber losses, it rests once a
target's sums hold two of those half-lives (200 stepped rows, a weight of 108, three quarters
of their limit) and the gains' mean square is at most their mean. Resting,
it takes no gains, so no row forms `|z|²` for it. It probes every 8th
row's gain, and wakes where the probe passes twice the resting mean, or
where a row's residual passes 8 of the target's spreads. On five
standardized features over 10 streams each, a rate increased after 2,000
rows from half the bound to 1.15 times it was first withheld within a row
of where an awake check withheld it (113 to 225 rows after the change).
From half the bound to 1.9 times, it was withheld within 8 rows of it (26
to 47 rows after). From 0.3 to 1.5 times, it was withheld within 27 rows
(65 to 99 rows after, the awake check's range). At a constant rate of 0.01
on 20 features, the check rests on 95% of the
rows. The Poisson and logistic checks take every gain, since their
residual is not in the target's units.

**`sgd` standardizes its features by default, as `kalman` does, because one
learning rate has to suit every feature.** Without it, features 100 times
larger put the default rate past its stability bound, so every prediction
is withheld as `unstable_rate`. Once the scaler has learned 22 rows by
Kish's effective sample size, the coefficients are held in the caller's
units, so the scaler's moving moves no prediction. Read through the
moments instead, a half-life of 10 put the slope at 3.03 for a truth of 2
at R² 0.978. Held in the caller's units, it is 2.02.

**The refit of each target's first rows brings the Huber loss's one pass
near statsmodels' `RLM`.** In one pass, the outliers capture the first
steps. [The refit of the first
rows](explanation/robust-scale.md#the-refit-of-the-first-rows) gives
`sgd(loss="huber")`'s figures over 40 seeds beside `huber`'s.

**Under `loss="quantile"` and `schedule="inv_scaling"`, `sgd` converges to
statsmodels' `QuantReg`.** At 100,000 rows, at the median and the 0.9
quantile of a skewed noise, its coefficients came within 0.022 and 0.027
of `QuantReg`'s, where the least-squares fit is 0.3 and 1.3 away.

**Under the squared loss, with `standardize` and no intercept, `sgd`
ends within 0.05 of numpy's least squares** over 20,000 rows, and the
coefficients it reports reproduce each prediction to 1e-13 of its size.

### `pa` — passive-aggressive regression

*API:* [`rheon.spec.pa`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.pa) — *Rust:* [`pa.rs`](../crates/rheon-core/src/pa.rs) — *Outputs:* [fields](OUTPUTS.md#pa)

Passive-aggressive regression, which moves the fit by the smallest step
that brings each row within a margin of its target.

Use `pa` to step on every row, in O(k) time a row for `k` features, without
tuning a learning rate. Use [`sgd`](#sgd--stochastic-gradient-descent) for
other losses or a learning rate, and
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) for an exact
fit on running sums.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.pa),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#pa). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes).

#### The update

Each row asks the fit to come within `eps·s_y` of its target, `s_y` the
target's own exponentially weighted standard deviation, as `sgd`'s tube is
drawn in. The step is the smallest change that does so. With `p` the
prediction, and the intercept's 1 inside `z`, a row takes the step its
`mode` sets:

```
loss = max(0, |y − p| − eps·s_y)      q = ‖z‖²
pa    τ = loss / q          pa1  τ = min(c, loss/q)      pa2  τ = loss / (q + 1/(2c))
β    += min(w, 1) · τ · sign(y − p) · z
```

That is the step under `standardize=False`, with `z = [1, x]`. Under
`standardize`, the default, the step is taken on the row standardized with
`sgd`'s scaler, `z̃`, which takes `z`'s place in `q` and the step. Past the
scaler's warm-up the coefficients are held in the caller's units, and the
step is mapped into them by the row's own means `mᵢ` and standard
deviations `sᵢ`:

```
step  = min(w, 1)·τ·sign(y − p)
βᵢ   += step·z̃ᵢ / sᵢ        β₀ += step − Σᵢ mᵢ·step·z̃ᵢ / sᵢ
```

The docstring's Notes, under *The margin's scale*, *c and the target's
units* and *The margin*, give why the margin is drawn in `s_y`, what `c`
is in, and how wide the margin is in units of the noise.

#### Example

**`pa` takes `sgd`'s bounds on the slopes.** Here the truth is
`1 + 2·x0 − x1`, and `coef_max=1.5` holds `x0`'s slope below its true
value, so the fit stays inside the bound, and its other coefficients move
to make up for it:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# y = 1 + 2·x0 − x1 plus noise.
rng = np.random.default_rng(7)
n = 20_000
x0, x1 = rng.standard_normal((2, n))
y = 1.0 + 2.0 * x0 - 1.0 * x1 + 0.3 * rng.standard_normal(n)
pl.DataFrame({"x0": x0, "x1": x1, "y": y}).write_parquet("pa.parquet")

# The same pa twice, the second with its slopes capped.
common = dict(targets=["y"], features=["x0", "x1"], half_life=float("inf"), c=0.1)
specs = [
    rp.spec.pa("free", **common),
    rp.spec.pa("capped", coef_max=1.5, **common),   # no slope above 1.5, so x0's cannot reach 2
]
bank = rp.ModelBank(specs)
bank.fit(pl.scan_parquet("pa.parquet"))
# What each fit found.
for name in ("free", "capped"):
    fitted = {r["term"]: round(r["coef"], 2) for r in bank.coef().filter(pl.col("spec") == name).iter_rows(named=True)}
    print(f"{name}: {fitted}")
```

```text
free: {'intercept': 0.99, 'x0': 2.02, 'x1': -0.98}
capped: {'intercept': 0.89, 'x0': 1.48, 'x1': -1.09}
```

The free fit finds the truth to within 0.02. The capped fit holds `x0`'s
slope under 1.5, and its intercept and `x1`'s slope move from the truth as
well, to 0.89 and −1.09.

#### See also

When `pa`'s steps are not the fit you need:

- [`sgd`](#sgd--stochastic-gradient-descent), gradient steps with a
  learning rate, under several losses, with the same scaler and bounds.
- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), an exact
  fit on running sums.
- [Follow a level that moves](how-to/follow-a-level-that-moves.md), to
  give a larger `c` or difference the target on a level that moves.
- [Choose a learning rate](how-to/choose-a-learning-rate.md#choose-pas-mode),
  to keep a `mode` that caps or damps the step where outliers are
  possible.
- [Bound the coefficients](how-to/bound-the-coefficients.md#bound-the-slopes-of-pa-with-a-small-c),
  to keep `c` small when `coef_min`, `coef_max` or `coef_sum` bound the
  slopes.

#### Notes

**With no margin, no intercept and unit weights, `pa` matches river's
`PARegressor`** in all three modes, on the raw features: the predictions
agree to 2e-15 and the coefficients to 9e-16 over 500 rows. The two are
compared at `eps=0` because river's margin is in the target's units, and
this model's is in units of `s_y`.

### `ftrl` — online logistic regression

*API:* [`rheon.spec.ftrl`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ftrl) — *Rust:* [`ftrl.rs`](../crates/rheon-core/src/ftrl.rs) — *Outputs:* [fields](OUTPUTS.md#ftrl)

Logistic regression for a 0/1 target, or a sparse linear regression, one
gradient step per row.

Use `ftrl` for a 0/1 target, or for a sparse fit that steps on every row
instead of solving. It fits by FTRL-Proximal (follow the regularized
leader, with a proximal step): its L1 penalty zeroes a coefficient with too little
evidence, and its rates adapt to each feature's history (McMahan et al.
2013). Use [`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs) for
an L1 penalty on the mean scale, which more rows or heavier ones do not
outweigh, and [`sgd`](#sgd--stochastic-gradient-descent) for other losses.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ftrl),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#ftrl). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes).

#### The update

`β` is the coefficients, and `zz`, `n` and `d` are the sums kept per
coefficient `i`. `λ_t` is the decay over the clock since the last row the
target was learned from, which is the page's `λ` on a stream where the
target is learned from every row (every row has it, at a weight above 0). `alpha`, `beta`, `l1`
and `l2` are the parameters of those names:

```
β_i  = 0 if |zz_i| ≤ l1 else −(zz_i − sign(zz_i)·l1) / (beta/alpha + d_i + l2)
p    = sigmoid(zᵀβ)                  zᵀβ itself under loss="squared"
zz_i ← λ_t·zz_i    n_i ← λ_t·n_i    d_i ← λ_t·d_i    on a row the target is learned from
g_i  = (p − y)·z_i·w      s_i = (√(n_i + g_i²) − √n_i)/alpha
zz_i += g_i − s_i·β_i      n_i += g_i²      d_i += s_i
```

Without a half-life, `d_i` is `√n_i/alpha`, so the denominator is
`(beta + √n_i)/alpha + l2`, the closed form of river's `FTRLProximal`.

Under `standardize=True` the same update runs on `z` standardized against
the features' running moments with the row admitted, as `sgd`'s is, so
`l1` zeroes a coefficient by its evidence in units of its feature's spread
rather than in the feature's own units. The fit stays in those
coordinates, and `coef` is read out in the features' own units with the
moments as they stand.

#### Example

**`l1` keeps the features with evidence and zeroes the rest.** Here a 0/1
target depends on 5 of 50 features. The fit keeps those five, and its
probabilities class the rows of the second half, each before it is learned:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# A 0/1 target from 50 features, of which only x0 to x4 move its probability.
rng = np.random.default_rng(8)
n, k = 20_000, 50
x = rng.standard_normal((n, k))
logit = x[:, :5] @ np.array([1.5, -1.0, 1.0, 0.8, -0.8])
up = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
names = [f"x{i}" for i in range(k)]
pl.DataFrame(x, schema=names).with_columns(up=up).write_parquet("clicks.parquet")

# One ftrl over all 50 features, each row predicted before it is learned.
sparse = rp.spec.ftrl(
    "sparse", targets=["up"], features=names, half_life=float("inf"),
    l1=300.0,                      # zero a coefficient whose summed evidence stays under 300
    strict_binary=True,            # refuse a target that is not 0 or 1
)
bank = rp.ModelBank([sparse])
out = pl.concat(bank.fit_predict_batches(pl.scan_parquet("clicks.parquet")))
# The features the L1 penalty kept.
slopes = bank.coef().filter(pl.col("term") != "intercept")
kept = slopes.filter(pl.col("coef") != 0.0)["term"].to_list()
# How often the second half's rows are classed right, out-of-sample.
p_up = out["sparse"].struct.field("pred_up")      # each row's probability, from the fit before the row
hits = ((p_up > 0.5).cast(pl.Float64) == out["up"])[n // 2:].mean()
print(f"coefficients left nonzero: {len(kept)} of {slopes.height}: {kept}")
print(f"rows of the second half classed right, out-of-sample: {hits:.2f}")
```

```text
coefficients left nonzero: 5 of 50: ['x0', 'x1', 'x2', 'x3', 'x4']
rows of the second half classed right, out-of-sample: 0.81
```

The penalty kept exactly the five features that move the probability, and
zeroed the other 45.

#### See also

When `ftrl`'s fit is not the one you need:

- [`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs), an L1
  penalty on the mean scale, solved on running sums.
- [`sgd`](#sgd--stochastic-gradient-descent), gradient steps under other
  losses.
- [Choose `ftrl`'s penalties](how-to/choose-ftrl-penalties.md), to work
  out how hard the penalties shrink the fit under a half-life.
- [Choose a learning rate](how-to/choose-a-learning-rate.md#choose-ftrls-alpha),
  to choose `alpha`, the pace at which it follows a break.
- [Scale a feature by its spread](how-to/scale-a-feature-by-its-spread.md),
  to give it features near 0, since it does not center them.

#### Notes

**With decay off, `ftrl` matches river's `FTRLProximal`:** the two agree to 1e-12, row for
row, and Vowpal Wabbit's `--ftrl` agrees to its single precision.

### `glm` — generalized linear models: Poisson, Gamma, binomial, Tweedie

*API:* [`rheon.spec.glm`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.glm) — *Rust:* [`glm.rs`](../crates/rheon-core/src/glm.rs) — *Outputs:* [fields](OUTPUTS.md#glm)

A generalized linear model, fitted one row at a time by one step of
iteratively reweighted least squares (Fisher scoring) a row.

Use `glm` for a target whose spread grows with its mean, or that has a
floor or a ceiling. Counts take `family="poisson"`, and positive amounts
such as prices, sizes and durations take `"gamma"`. Proportions and 0/1
outcomes take `"binomial"`, and amounts with an exact zero, such as an
insurance claim, take `"tweedie"`. Its prediction is the mean on the target's own scale. It
takes O(k³) time a row and target for `k` coefficients, a Cholesky
factorization; [`sgd`](#sgd--stochastic-gradient-descent) with
`loss="poisson"` or `"logistic"` takes a gradient step on the same
likelihood in O(k).

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.glm),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#glm). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes).

#### The update

Each target keeps a Gram `G` and its coefficients `β`. With `η = zᵀβ` the
fit before the row, `μ = g⁻¹(η)` its mean through the link `g`, `V` the
family's variance and `R` the ridge:

```
W  = 1 / (g'(μ)² V(μ))          z̃ = η + (y − μ) g'(μ)
G  = λ·G + w·W·z zᵀ
β += (G + R·I)⁻¹ · w·W·(z̃ − η)·z
```

`pred` is `μ`, made before the row is learned. Without decay this is the
penalized batch fit's IRLS, the rows' weights `W` frozen at the fit that
scored them, so it tracks statsmodels' `GLM` with `freq_weights` on the
same rows. The bias the frozen weights leave shrinks as `1/n`, against
the standard error's `1/√n`. A family's links are its canonical one and the
log: the Poisson's log, the Gamma's log and inverse, the binomial's logit,
and the Tweedie's log and power `μ^(1 − p)`.

**Under the inverse and power links each row is fitted exactly.** These
links give a mean only where `η > 0`. A Fisher step scores a row at the fit
before it, so a target far above its mean, such as a Gamma amount 1000
times it, pushes `β` past the domain's edge for the rows around it. Those
rows then have no mean and teach nothing, so none brings the fit back.
Once a target's refit has run, each row instead minimises its own
penalized deviance exactly, `w·d(y, μ) / 2` plus the penalty
`(b − β)ᵀ(G + R·I)(b − β) / 2`. With `P = (G + R·I)⁻¹` and `q = zᵀP z`,
the row's `η` moves to the one root `u > 0` of
`u − η + (w·q / (p − 1))·(y − u^(−1/(p − 1)))`, with `p = 2` for the Gamma:

```
β += ((u − η) / q) · P·z
G  = λ·G + w·min(W(η), W(u))·z zᵀ
```

The root lies between `η` and the link at `y`, so the step never passes the
target, and it is above 0 from any `η`. A refit given up past
`refit_budget` starts from every coefficient at 0 in the same way.
`step_cap` holds `|u − η|`. The Gamma's root has a closed form; the
Tweedie's takes two or three Newton passes a row.

Measured against statsmodels' `GLM` on 16 adversarial streams of 3,000
rows, 20 seeds each, the gap is in statsmodels' standard errors, after the
default refit. The streams without a far target include `η` near the
domain's edge, a rare category near 0, rows sorted by a feature, heavy
noise, a misspecified mean and zero weights. On them the median seed was
0.03 to 0.21 off (0.58 at 20 features and 10,000 rows) and the worst 0.11
to 0.86. On the streams with one target 1000 times its mean, or a first
target of 1e6, the median seed was 0.8 to 1.6 off and the worst 6.2, with
at most 16 null predictions after the refit. The Fisher step on the same
streams left the worst seed 58 standard errors off, with up to 933 null
predictions.

**Under the inverse and power links the ridge depends on the target's
units.** The coefficients are on the scale of `1/μ` (or `μ^(1 − p)`), so a
target measured in units a hundredth as large has coefficients a hundred
times as large, which the same `ridge` pulls toward 0 far harder. On the
base stream above, at the default `ridge` of 1 and 3,000 rows, the fit sat
0.06 standard errors from statsmodels' unpenalized fit with an intercept
`η` near 0.5, 1.1 with it near 5, and 36 with it near 50. Under the log
link a change of units moves the intercept by its log alone. Scale such a
target so that `η` is of order 1, or lower `ridge`.

**The first rows are refit.** Before `k` rows the fit is underdetermined,
and one row can throw it off for good: a Gamma row scored at a very
negative `η` steps by `y·e^|η|`, and a Poisson row scored at too large a
mean starves its direction of curvature. Each target's first `refit_rows`
learned rows are therefore kept and, once the buffer is full, fitted to
convergence by penalized IRLS, and `G` is rebuilt at that fit; the buffer
is then freed. While it fills, each row's own step in `η` is held to 1, so
the predictions before the refit are sound. `step_cap` instead holds each
row's own step in `η` to a bound, with no buffer, at a larger bias without
decay.

**Under the log link a step stops at the row's own target.** An upward
step moves `η` by an amount that grows as `y μ^(1−p)` as the mean falls,
`p` the variance power. So with neither a refit nor a cap, or after the
default refit, where the cap is off, one far target can send the
coefficients away. Without either remedy that happened on 17 of 100
seeded streams of 300 rows for the Gamma and on 64 for the Tweedie at `p =
3`. Under the default spec, one target 1000 times its draw did it on 14 to
48 of 100. On each row the start's cap does not hold, a step that would
carry `η` past `ln y` is scaled as `step_cap` scales one. It lands on the
root in `(η, ln y]` of the row's own penalized deviance, and none of those
seeds runs away. Rows it does not fire on take the step unchanged. It is a
floor, not a remedy: the refit, or a cap, is what makes the fit accurate.

**The spans.** `half_life`, `gap_cap` and `refit_span` take a number (rows,
with no clock) or a clock duration, as every spec's do. `refit_rows` counts
rows of positive weight. `refit_span` is the same refit over a span of the
clock: a target's rows from its first learned row through the first row at
least `refit_span` after it, once they hold a row per coefficient. The span
is read off the rows' exact stamps. On a clock of one unit a row that is
`refit_span + 1` rows, the refit of `refit_rows = refit_span + 1` to the
bit. Both are held to `max_state_bytes`, 256 KB a group by default and at
most 64 MiB. A count's buffer has a size known when the spec is built, and
is refused there past it. A span's holds the rows the data puts in it, so
it is held to it as the rows come. A group past half of it gets a warning,
once, and the row that would pass it stops the run with an error naming
the group and the bytes.

#### Example

**`glm` predicts a count's mean on the count's scale.** Here the truth is
a Poisson count whose log mean is `0.5 + 0.3·x`, and the coefficients are
on the log scale:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# Poisson counts with a log mean of 0.5 + 0.3 x.
rng = np.random.default_rng(5)
x = rng.standard_normal(20_000)
y = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
pl.DataFrame({"x": x, "y": y}).write_parquet("counts.parquet")

# One Poisson fit, with no decay, learned over the file one chunk at a time.
bank = rp.ModelBank([rp.spec.glm("pois", targets=["y"], features=["x"], half_life=float("inf"))])
bank.fit(pl.scan_parquet("counts.parquet"))
# The coefficients are on the log scale.
print({r["term"]: round(r["coef"], 2) for r in bank.coef().iter_rows(named=True)})
```

```text
{'intercept': 0.5, 'x': 0.3}
```

#### See also

When `glm`'s fit is not the one you need:

- [`sgd`](#sgd--stochastic-gradient-descent) with `loss="poisson"` or
  `"logistic"`, a gradient step on the same likelihood in O(k) time.
- [`ftrl`](#ftrl--online-logistic-regression), a sparse logistic
  regression.
- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), the
  least-squares fit, for a target whose spread does not grow with its
  mean.

#### Notes

**Measured against statsmodels and glum** over 50 seeds at 20
coefficients, the ridge matched. The largest gap was 0.47, 0.33 and 0.61
standard errors at 10,000 rows (Poisson; Gamma of shape 2; Tweedie at
`p = 1.5`, dispersion 2), and 0.20, 0.16 and 0.31 at 100,000. The fit's
error against the true coefficients equals the batch fit's to 0.01
standard errors on average.

**The state is bounded by the spec.** A target keeps `k(k+1)/2 + k + 1`
numbers, and until its refit runs a buffer of `refit_rows·(features + 2)`,
the one part whose size is not the model's shape. A group's buffers are
held to `max_state_bytes` (256 KB by default) when the spec is built: a
`refit_rows` past it is refused, naming the bytes, and the default shrinks
to fit.

**There is no window.** A row's contribution to `G` is its weight at the
fit that scored it, so removing a row exactly needs the row itself. A
window of 2,000 rows at 20 coefficients would hold 384 KB a target, past a
group's cap; decay forgets in place.

### `holt` — Holt's linear trend

*API:* [`rheon.spec.holt`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.holt) — *Rust:* [`holt.rs`](../crates/rheon-core/src/holt.rs) — *Outputs:* [fields](OUTPUTS.md#holt)

Holt's linear trend, which extrapolates the target's own level and slope.

Use `holt` as the baseline a model with features should beat: it is the
only linear model that takes no features. Run it in the same bank as the
regression, with `emit_sigma=True` on both, then compare the two models'
`sigma` ([How wide is its
error?](DIAGNOSTICS.md#how-wide-is-its-error)), or let a
[`seqtest`](#seqtest--a-sequential-test-of-a-sign-by-betting) with `a` and
`b` say which predicts closer.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.holt),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#holt). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes),
apart from `features`, which `holt` does not take.


**Forecasts at horizons.** `forecast_spans` asks for the forecast a span `Δ` of
the clock past the row and its integral over the span. Both read the level
and trend before the row: `forecast_<h>_<t> = l + b (s + Δ)` and
`forecast_sum_<h>_<t> = Δ (l + b (s + Δ/2))`. At `Δ = 0` the forecast is
`pred_<t>`. A span is a number of clock units, rows without a clock, or a
duration such as `"30d"`, never a count of rows: the trend is per clock unit
and the clock of rows to come is not known. No forecast variance is given.
The variance of exponential smoothing's forecast (with additive errors
and trend, `statsmodels`' `ETSModel`) counts the innovations of the rows
within the span and assumes fixed gains. Here that holds only on evenly
spaced rows of equal weight once the gains have stopped moving. There it
matched the forecast errors' variance within 0.7% at 1 to 10 rows ahead,
over 195,000 simulated rows. On a clock the rows a span holds are the data's
to decide, and the model keeps no noise variance of its own.

#### The update

Each row moves the level `l` and the trend `b` by a gain times the
residual `e = y − pred`. `s` is the clock since the target was last
observed, this row's step included, each step capped at `gap_cap` (or
`session_gap` at a session change). So a row with a null target or weight
0 leaves the level and trend where they were, and its step goes into the
next observation's `s`. `W` and `V` are the weight the level and the trend
have gathered, and `S0`, `ā` and `Q` are the line fit's weight, mean age
and spread of ages. A row takes the larger of two gains, on the level and
on the slope:

```
λ_l = 0.5^(s/half_life)                        λ_b = 0.5^(s/trend_half_life)
pred = l + b·s                                 e = y − pred
textbook:  g_t = w/(λ_l·W + w)                 W' = λ_l·W + w
           h_t = w/(λ_b·V + w) · g/s           V' = λ_b·V + w     (s > 0)
line fit:  S0' = S0 + w     ā' = S0·(ā + s)/S0'     Q' = Q + S0·w·(ā + s)²/S0'
           g_f = w·(1/S0' + ā'²/Q')            h_f = w·ā'/Q'
l' = pred + g·e     g = g_f under half_life=inf, else max(g_f, g_t)
b' = b + h·e        h = h_f under trend_half_life=inf or at s = 0, else max(h_f, h_t)
```

Under `trend=False` the line fit is not used: `g = g_t` at every
half-life, and `b = 0`. A target's first observation sets `l = y`,
predicts nothing, and starts the trend at 0. The docstring's Notes, under
*The textbook gains*, *The line fit* and *Which gain a row takes*, give
what each gain is and when each wins.

#### Seasons, damping and the starts

`seasonal="add"` or `"mul"` adds a season of `seasonal_periods` values `S`
over `period` clock units, and `damping_half_life` damps the trend.
`start` says where the fit starts. `"known"` starts from the initial states
given. `"heuristic"` fits them to a target's first rows (10, or 5 seasons)
as statsmodels' heuristic does, then learns those rows from them, and
withholds them as `below_warm_rows`. A season or a damped trend needs one of these two
starts; `"warm"` is the start above. Under them every gain is the fixed
one, `α = 1 − λ_l` at weight 1, and the update is the Holt-Winters
recursion, `p` the row's phase and `L` the level carried over the step:

```
φ_s = φ·(1 − φ^s)/(1 − φ)  (s undamped)        φ = 0.5^(1/damping_half_life)
L = l + φ_s·b                                   pred = L + S[p]  or  L·S[p]
l' = α·y − α·S[p] + (1 − α)·L    (add)          l' = α·y/S[p] + (1 − α)·L    (mul)
b' = β·(l' − l)/s + (1 − β)·φ^s·b
S'[p] = γ·y − γ·L + (1 − γ)·S[p] (add)          S'[p] = γ·y/L + (1 − γ)·S[p] (mul)
```

`β = 1 − λ_b`, and `γ = 1 − 0.5^(Δ/seasonal_half_life)` with `Δ` the clock
since phase `p` was last learned: the season's length on an even clock. A
row of weight `w` takes `w·α/(λ + w·α)` in place of each gain. On a row
clock at weight 1 this is statsmodels' `ExponentialSmoothing` with the same
fixed parameters, operation for operation, to the bit on Linux x86-64.

A row's phase is `⌊m·(c − c₀)/P⌋ mod m` of its own clock value `c`, `c₀`
the first row's: the raw value, which `gap_cap` and `session_gap` do not
shorten, in integers on a temporal or an integer clock. A null target keeps
its row's place in the season; without a clock column a row of weight 0
counts in no clock, so the row after it takes its place. Under `"mul"` a
row whose target, or whose level or factor after the update, would be 0 or
below is predicted, not learned, and its `withheld_reason` is
`target_not_positive`. The state grows by 16 bytes a seasonal value a target, and
the heuristic start's held rows by 36 bytes each until it is fitted;
`max_state_bytes` (256 KB by default) refuses a spec past it, naming the
bytes it needs.

#### Example

**A feature earns its place when the model that reads it beats `holt`.**
Here the target's level wanders and trends, and `2·x` rides on it. A
regression on `x`, whose intercept follows the level, beats `holt` on the
spread of its error, and `seqtest` says when the evidence for that is
enough:

```python
import math

import numpy as np
import polars as pl
from rheon import polars as rp

# A level that wanders and trends, plus 2·x and noise, one row a clock unit.
rng = np.random.default_rng(9)
n = 2_000
t = np.arange(n, dtype=float)
x = rng.standard_normal(n)
level = 0.02 * t + np.cumsum(0.1 * rng.standard_normal(n))
y = level + 2.0 * x + 0.5 * rng.standard_normal(n)
pl.DataFrame({"t": t, "x": x, "y": y}).write_parquet("trend.parquet")

# A regression on x, holt alone, and a test of which predicts closer.
common = dict(targets=["y"], clock="t", gap_cap=10.0, half_life=20.0, emit_sigma=True)   # sigma_y: the error's spread
specs = [
    rp.spec.ewridge("with_x", features=["x"], **common),    # the feature, and an intercept that follows the level
    rp.spec.holt("naive", **common),                         # the target's own level and trend alone
    rp.spec.seqtest("gain", targets=["y"], a="with_x", b="naive"),   # evidence that with_x predicts closer
]
out = pl.scan_parquet("trend.parquet").rheon.fit_predict(specs).collect()
# Each model's error spread, then when the evidence for with_x is enough.
for name in ("with_x", "naive"):
    print(f"{name}: sigma_y at the last row {out[name].struct.field('sigma_y')[-1]:.2f}")
passed = out["gain"].struct.field("log_e_a_y") >= math.log(20)   # the 5% level, read at any row
print(f"with_x predicts closer at the 5% level from row {passed.arg_true()[0]}")
```

```text
with_x: sigma_y at the last row 1.14
naive: sigma_y at the last row 2.21
with_x predicts closer at the 5% level from row 15
```

The regression's error spread is about half of `holt`'s, and the test
finds the evidence enough at row 15.

#### See also

When `holt` is not the baseline you need, or to test a model against it:

- [`seqtest`](#seqtest--a-sequential-test-of-a-sign-by-betting), to test
  whether a regression predicts closer than this baseline.
- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), a
  regression on features to set against it.
- [How wide is its error?](DIAGNOSTICS.md#how-wide-is-its-error), the
  `sigma` that compares the two.
- [Choose Holt's smoothing](how-to/choose-holts-smoothing.md), to set
  how fast the level and the trend forget, or a flat prediction.
- [Group on a column made in the
  query](how-to/fit-one-model-per-group.md#group-on-a-column-made-in-the-query),
  for a seasonal baseline, one level and trend per phase.

#### Notes

**The line fit keeps the trend from being pinned to the first
observation.** Gains of `1/n` on both level and trend, without the line
fit, pinned the trend to the first observation. The level then carried
that observation's noise times `ln n`: 11 off a flat series after 40,000
rows without decay, and 6.3 after 10,000 rows at `half_life=10000`, the
worst of 50 seeds at a noise of 0.5. The line fit's worst was 0.010 and
0.026, the weighted least squares' own.

**Without decay, `holt` fits the weighted least-squares line,** whatever the weights
and the clock: statsmodels' `WLS` on `(1, t)` agrees to 1e-15. Under a
half-life, every row takes the textbook's gains after a warm-up of a few
half-lives (from row 745 at half-lives of 6 and 25). From row 1,000 in
that run, where the textbook gains have reached statsmodels' fixed
ones to 1e-12, the fit is statsmodels' `Holt` to 1e-12.

### `ar` — an autoregression on the target's own past

*API:* [`rheon.spec.ar`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ar) — *Rust:* [`ar.rs`](../crates/rheon-core/src/ar.rs) — *Outputs:* [fields](OUTPUTS.md#ar)

An autoregression for each target, AR(p): the target regressed on its own
last values, and on the features where they are given, by exponentially
weighted least squares.

Use `ar` to forecast a series one row ahead from its recent past, without
building lag columns in the query: the model keeps each target's last
values itself. With no decay its coefficients are statsmodels' `AutoReg`'s
on the same rows; with a `half_life` it follows dynamics that drift. Each
target is its own series, with its own lags and fit.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ar),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#ar). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes);
`features` is optional.

#### The update

For lags `ℓ_1 < … < ℓ_q`, row `t`'s design is
`z_t = (y_{t−ℓ_1}, …, y_{t−ℓ_q}, x_t)`, its lags read from the target's last
learned values. `W_y` is the target's own weight, and `S`, `r` its running
means of `z zᵀ` and `z·y`, kept centered:

```
pred_t = c + β·z_t                       from the fit before the row
W_y'   = λ·W_y + w                       a learned row of the target only
S'     = (λ·W_y·S + w·z zᵀ)/W_y'          r' = (λ·W_y·r + w·z·y)/W_y'
β      = (S + ridge·I)⁻¹ r                c = ȳ − β·z̄     after every learned row
```

A learned row of a target has a weight above 0 and the target present. It
joins the target's last values, and enters `S` and `r` once `max(lags)`
values precede it. A row of weight 0, or with the target null, is not a
lag: the next row's lags are those before it. A lag counts learned rows,
not clock units, and a session change or a gap past `gap_cap` empties the
lags, keeping the fit. A target predicts once its Gram plus the ridge
is positive definite, every Cholesky pivot above `1e-10` of its diagonal: the guard on
the early rows is that condition, not a count of rows, and `min_weight`
defaults to 0.

#### Example

**An AR(2) found from the series alone.** Here `y` follows
`2 + 0.6·y₋₁ − 0.3·y₋₂` plus noise:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# y = 2 + 0.6 y[-1] - 0.3 y[-2] + noise, one row a clock unit.
rng = np.random.default_rng(3)
y = [5.0, 5.0]
for e in rng.standard_normal(20_000):
    y.append(2.0 + 0.6 * y[-1] - 0.3 * y[-2] + e)
pl.DataFrame({"y": y}).write_parquet("series.parquet")

# Two lags and a constant, from the whole history.
bank = rp.ModelBank([rp.spec.ar("a", targets=["y"], lags=2, half_life=float("inf"))])
bank.fit(pl.scan_parquet("series.parquet"))
for row in bank.coef().iter_rows(named=True):
    print(f"{row['term']}: {row['coef']:.3f}")
```

```text
intercept: 2.022
lag1: 0.603
lag2: -0.309
```

The fit lands within sampling error of `(2, 0.6, −0.3)` after 20,000 rows.

**A trend or a season comes in as features.** `ar` has no `trend` or
`seasonal` parameter of its own: a model sees the clock with every gap past
`gap_cap` shortened, so a time or a phase read from it would drift after
each long gap. Make them as columns in the query, from the clock column as
it is, and pass them in `features`. Here an hourly series has a slow trend
and a daily cycle on top of its AR(1) noise:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# y = 10 + 0.001 t + 2 sin(2 pi t / 24) + e, e an AR(1) of 0.6, one row an hour.
rng = np.random.default_rng(5)
n = 24 * 400
t = np.arange(n, dtype=float)
e = np.zeros(n)
for i in range(1, n):
    e[i] = 0.6 * e[i - 1] + rng.standard_normal()
y = 10.0 + 0.001 * t + 2.0 * np.sin(2 * np.pi * t / 24) + e
pl.DataFrame({"t": t, "y": y}).write_parquet("hourly.parquet")

# The time and the day's phase, made in the query.
phase = 2 * np.pi * pl.col("t") / 24
lf = pl.scan_parquet("hourly.parquet").with_columns(
    trend=pl.col("t"), day_sin=phase.sin(), day_cos=phase.cos()
)
spec = rp.spec.ar(
    "a", targets=["y"], lags=1, features=["trend", "day_sin", "day_cos"],
    clock="t", gap_cap=24.0, half_life=float("inf"),
)
bank = rp.ModelBank([spec])
bank.fit(lf)
for row in bank.coef().iter_rows(named=True):
    print(f"{row['term']}: {row['coef']:.4g}")
```

```text
intercept: 3.983
lag1: 0.6042
trend: 0.0003951
day_sin: 0.8368
day_cos: 0.3044
```

The lag of `y` carries part of the trend and the cycle, so their
coefficients are the generating ones seen through it. The trend's is
`0.001 · (1 − 0.6)`. The cycle's are `2 (1 − 0.6 cos(2π/24))` and
`2 · 0.6 sin(2π/24)`, about 0.84 and 0.31.

#### See also

When the lags are not of the target, or the question is not its value:

- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), the same
  least squares on lag columns made in the query, for lags across columns.
- [`holt`](#holt--holts-linear-trend), a level and a trend rather than a
  regression on lags.
- [`garch`](#garch--how-large-will-the-next-return-be), for the variance
  of a series rather than its value.

#### Notes

**With no decay, `ar` is `AutoReg`.** Its coefficients are least squares on
the rows after the first `max(lags)`, which is what
`AutoReg(y, lags, trend="c", exog=x).fit()` computes: within 1e-9 relative
over 5,000 rows, at orders 2 to 5 and at lags `[1, 2, 12]`. A null target
is `AutoReg(missing="drop")`. Under a half-life the coefficients are
statsmodels' `WLS` on the lag design, each row weighted
`w·0.5^(age/half_life)` on the clock, within 1e-9 as well.

**`select` chooses the order as `ar_select_order` does.** Every order is
read from the one fit at the largest: with the features first and the lags
after, an order's system is a leading block of the largest one's, so one
Cholesky factor gives every order's residuals. The count in AIC and BIC is
Kish's effective number of rows, the row count with no decay. With no decay it
chose `ar_select_order`'s order on all twelve series of true orders 1 to 4
it was checked on, by AIC and by BIC, with a feature and without. Under
decay it chose the order that minimizes the criterion over statsmodels'
`WLS` per order, on all six series checked.

### `isotonic` — a monotone function of one feature

*API:* [`rheon.spec.isotonic`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.isotonic) — *Rust:* [`isotonic.rs`](../crates/rheon-core/src/isotonic.rs) — *Outputs:* [fields](OUTPUTS.md#isotonic)

Isotonic regression of each target on one feature: the best function of the
feature that never falls as it rises (or never rises, with
`increasing=False`), fitted to the rows before each row and read at the
row's own feature value.

Use `isotonic` as a monotone calibrator, turning a score into a probability
or a ranking into an expected value, or wherever a target should only move
one way with a feature and nothing else is assumed of the shape. The fit is
scikit-learn's `IsotonicRegression` on the same rows and weights: over every
row, over a window of rows or of the clock, or on rows weighed by a decay.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.isotonic),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#isotonic). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes);
`features` names exactly one column.

#### The update

For target `t` and the rows `(x_i, y_i, w_i)` learned before the row, each
weight times its decay `d_i` (1 without one):

```text
f_t  = argmin_f  sum_i w_i d_i (y_i - f(x_i))^2   subject to  f non-decreasing in x
f_t  <- min(max(f_t, y_min), y_max)
pred = f_t at the row's x, on the line between the two nearest x fitted
```

Rows at one feature value are pooled first, their weights summed and their
value the weighted mean. Pool-adjacent-violators then walks the distinct
values in order. It merges each run of neighbors that breaks the order into
one block at its weighted mean, and goes back over earlier blocks as a merge
lowers them; the result is the unique solution of the problem above. The
model runs scipy's implementation of it line for line, then reads the fit as
`numpy.interp` does. Beyond the range of the rows learned the prediction is
null (`out_of_bounds="nan"`, scikit-learn's default, withheld as
`not_identified`) or the nearer end's value (`"clip"`); one distinct feature
value is its value everywhere.

**Every learned row refits in full.** A block of the old fit that does not
hold the new row never splits, so a local refit would do, but its block
means would be summed in another order than scipy's. The whole refit costs
time in proportion to the distinct feature values held, `m`: about 2 ns a
value on arm64 (macOS, release build, one target), so 0.3 µs a row at 100
values, 2.0 µs at 1,000 and 16 µs at 8,000. Under a window it is about
3.4 ns a row the window holds, since its rows at one value are pooled at
each refit.

**The state** is, a target, each distinct feature value once with its weight
and its weighted sum, or under a window each row in it, sorted by feature
and target value. Neither is bounded by the settings, so it is checked
against `max_state_bytes` as the rows go in, and a run that would pass it
stops naming the group. A continuous score never repeats, so round it in
the query: `pl.col("score").round(3)` keeps at most 1,001 values between 0
and 1, and the fit is then exactly scikit-learn's on the rounded score. A
decay does not bound it either, since an old row's weight shrinks but stays
in the fit; a window does.

**A decay** keeps each row's weight scaled from an origin on the clock,
`w / decay(t_i - t_0)`, since the fit does not move when every weight is
scaled alike: no weight is touched on a row, and the origin moves every 128
half-lives.

**Against scikit-learn.** With no decay and no repeated feature value
without a window, or under any window, each prediction is a longhand
transcription of what scikit-learn runs, to the bit. In turn, scikit-learn's
own prediction is that transcription to the bit on every fit too, unfused, or
with each multiply and add fused into one rounding as the arm64 wheels of
scipy and numpy compute them. So the only gap is that rounding: 8.3e-16
at worst on targets of order 1, over ten seeds of six forms (both
directions, both bounds, a row window and a clock window under either
edge). A feature value repeated without a window is pooled as its rows
arrive, and a decayed weight is one `exp2` of the clock, where scikit-learn
is handed the decay factor: 2.1e-15 at worst on the same targets.

### `vector_ar` — a vector autoregression with Granger tests

*API:* [`rheon.spec.vector_ar`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.vector_ar) — *Rust:* [`vector_ar.rs`](../crates/rheon-core/src/vector_ar.rs) — *Outputs:* [fields](OUTPUTS.md#vector_ar)

A vector autoregression, VAR(p), of the targets together: each target
regressed on the last values of every target, and on the features where
they are given, by exponentially weighted least squares, with Granger
causality tests reported on every row.

Use `vector_ar` when one series' past predicts another's, or when the
question is whether it does. Each row carries the declared tests'
statistics and p-values, and the bank reads the impulse responses, their
standard errors, the forecast-error variance decomposition, the roots, the
long-run mean and every order's criteria from the fit
([`ModelBank.vector_ar`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.vector_ar)).
With no decay its coefficients, standard errors and tests are statsmodels'
`VAR(y, exog).fit(p)`'s on the same rows.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.vector_ar),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#vector_ar). The
parameters every spec takes are in [What every spec takes](#what-every-spec-takes);
`features` is optional.

#### The update

Row `t`'s design is `z_t = (x_t, y_{t−1}, …, y_{t−p})`, `d = e + kp`
numbers for `e` features and `k` targets, its lags read from the targets'
last learned values. Every equation has the same inputs, so least squares
equation by equation is the system's fit. One set of running means serves
them all: `S_zz`, `S_zy` and `S_yy`, the means of `z zᵀ`, `z yᵀ` and
`y yᵀ` over the learned rows, kept centered:

```
pred_t = c + Bᵀ z_t                       from the fit before the row
W_y'   = λ·W_y + w                        a learned row only
M'     = (λ·W_y·M + w·v vᵀ)/W_y'           v = (z_t, y_t), M holding the three
B      = (S_zz + ridge·I)⁻¹ S_zy           c = ȳ − Bᵀ z̄     at each solve
SSE    = S_yy − S_zyᵀ B                    the residual co-moments
```

A learned row has a weight above 0 and every target present. It joins the
last values and enters `M` once `p` rows precede it. A row of weight 0, or
with a target null, is not a lag; the row with a target null is still
predicted, every target. A lag counts learned rows, not clock units, and a
session change or a gap past `gap_cap` empties the lags, keeping the fit.
The fit is read once `S_zz` plus the ridge is positive definite, every
Cholesky pivot above `1e-10` of its diagonal; before that the predictions
are withheld as `not_identified`. The solve runs on `ewridge`'s schedule,
and at each solve one Cholesky factor `S_zz = L Lᵀ` gives the tests and
the standard errors. With `n = W_y² / Σw²`, Kish's count, `df = n − (d+1)`
and `Σ_u = SSE·n/df`, a test of the lags of the causing series `J` in the
equations `E` is

```
λ = df · tr(SSE_E⁻¹ βᵀ G⁻¹ β)              G = (S_zz⁻¹)_RR,  β = B[R, E]
```

with `R` the lag columns of `J`: the Wald statistic of the restrictions,
with no restricted fit run.

#### Example

**Which series leads.** Here `a` drives `b` one row later, and `b` does
not drive `a`:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# a = 0.5 a[-1] + e, b = 0.3 b[-1] + 0.4 a[-1] + e.
rng = np.random.default_rng(4)
a, b = [0.0], [0.0]
for _ in range(5_000):
    a.append(0.5 * a[-1] + rng.standard_normal())
    b.append(0.3 * b[-1] + 0.4 * a[-2] + rng.standard_normal())
df = pl.DataFrame({"a": a, "b": b})

# Two lags of each series, a test of each on the other, no decay.
spec = rp.spec.vector_ar(
    "v", targets=["a", "b"], lags=2, half_life=float("inf"),
    causality=[{"caused": "b", "causing": "a"}, {"caused": "a", "causing": "b"}],
)
bank = rp.ModelBank([spec])

# The tests on the last row, then b's response to a shock to a.
last = bank.fit_predict(df)["v"][-1]
print(f"a -> b: p = {last['causality0_p']:.0e}")
print(f"b -> a: p = {last['causality1_p']:.2f}")
irf = bank.vector_ar(table="irf", periods=2).filter(response="b", impulse="a")
print([(p, round(v, 2)) for p, v in irf.select("period", "value").rows()])
```

```text
a -> b: p = 1e-190
b -> a: p = 0.98
[(0, 0.0), (1, 0.38), (2, 0.32)]
```

The test of `a` in `b`'s equation rejects at any level, and that of `b` in
`a`'s does not. A unit shock to `a` moves `b` by about 0.4 a row later,
and by `0.4 (0.5 + 0.3)` a row after that.

#### See also

When the targets do not move together, or the lags are not of the targets:

- [`ar`](#ar--an-autoregression-on-the-targets-own-past), each target on
  its own past alone.
- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), least
  squares on lag columns made in the query.

#### Notes

**`ar` or `vector_ar`.** Use `ar` when each series is predicted from its
own past, when the series are missing on different rows (each of `ar`'s
targets keeps its own series), or when there are many series: its cost a
row is `k (p + e)²`. Use `vector_ar` when one series' past predicts
another's, when the question is Granger causality, the impulse responses
or the variance decomposition, and when the series arrive together: its
cost a row is `(kp + e + k)²` on one set of running means. A row missing
any series teaches `vector_ar` nothing. With one target and no features
the two are the same fit, to rounding.

**With no decay, `vector_ar` is statsmodels' `VAR`.** On the test suite's
streams the coefficients are within 1e-12 of `VAR(y, exog).fit(p).params`'s
largest and the standard errors within 1e-11 of its `bse`. The statistics
of `test_causality` and `grangercausalitytests` are within 1e-11 and their
p-values within 1e-10, `select_order`'s criteria within 1e-13, and the
impulse responses, their standard errors and `fevd` within 1e-10. Under a
half-life the coefficients are statsmodels' `WLS` per equation, each row
weighted `w·0.5^(age/half_life)` on the clock, within 1e-11.

**The tests under decay.** Kish's count stands for the rows: the
coefficients' covariance is `(S_zz⁻¹/n) ⊗ Σ_u`, statsmodels' at no decay.
It was measured on a VAR(1) of two series with no causation, read at the
end of 1,000 streams of 2,000 rows. The F test at 5% rejected 4.4% of the
time at `λ = 0.99`, about 199 effective rows, and 3.6% at `λ = 0.95`, about
39. So it holds its size once the memory spans a few hundred rows, and is
conservative below that.

**The state** is the lags (`8kp` bytes), the running means over `d + k + 1`
slots and what the last solve read, set by the settings and never by the
stream's length. It is checked against `max_state_bytes` when the spec is
built: the default 256 KB takes up to 41 lags of 2 targets, 15 of 5 and 7
of 10, and a spec past it is refused with the bytes it would need.

## Moments and correlation

These models measure the columns you name without fitting a target: their
means, variances and correlations, for a few columns or for thousands, and,
in `audit`'s case, the counts of what the columns hold. `ew_cov`,
`ewm_moments` and `deco` write their statistics on every row, from the state
before that row. The
means, spreads, correlations and `rho` read that state alone, so they can
serve as features for the row without leaking it. `mahal`, the
principal-component scores, `u` and `loglik` measure the row's own values
against that state. `marginal`,
`rcov` and `audit` write only `weight_sum` on a row, and keep their results
in the state: read them as a table after the run, or, for `rcov`, when a
block closes. `garch` forecasts each return's variance before the row
arrives, from a model of how variance clusters in time, and `dcc` the
correlations of several returns from a model of how they move together.

### `ew_cov` — exponentially weighted moments

*API:* [`rheon.spec.ew_cov`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ew_cov) — *Rust:* [`ewcov.rs`](../crates/rheon-core/src/ewcov.rs) — *Outputs:* [fields](OUTPUTS.md#ew_cov)

Exponentially weighted moments of the feature columns, and the statistics
read off them.

Use `ew_cov` for the running means, spreads, correlations, principal
components or Mahalanobis distances of a stream's columns, each read before
its row so that it can be a feature for that row. It takes one O(k²) step a
row for `k` columns, where Polars expressions alone take O(k²) *passes over
the data*, one for each pair's exponentially weighted correlation. Use
[`marginal`](#marginal--every-pairs-moments-kept-in-the-state) to screen
thousands of features against a few targets,
[`deco`](#deco--one-correlation-for-the-whole-matrix) for one number that
moves with the whole correlation matrix, and
[`rcov`](#rcov--a-blocks-realized-covariance-robust-to-noise) for a block's
covariance under microstructure noise.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ew_cov),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#ew_cov). The
parameters every spec takes are in [What every spec
takes](#what-every-spec-takes).

#### The update

The model keeps `m`, the weighted mean of the row's features `x`, and `C`,
the weighted mean of `(x − m)(x − m)ᵀ`, beside `W` and `Q`, the sum of the
squared weights. It updates them in Welford's centered form, because the
raw form, `E[x xᵀ] − m mᵀ`, loses a variance when the columns sit far from
zero. `c` is Bessel's correction at Kish's effective number of rows,
`n = W² / Q`, as Polars' `ewm_var` applies it under its default,
`bias=False`. Give `bias=True` for `C` itself, Polars' `bias=True` and
`ddof=0`. No correlation moves, since `c` cancels from each. `P`, read
before the row, serves `partial_corr` and `mahal` alone, with `s` a scale
on `precision_prior` that decays with the co-moments. Each lag `ℓ` of
`lags` keeps `C_ℓ`, with both deviations taken against the mean before the
row:

```
W' = λW + w      Q' = λ²Q + w²      a = λW / W'      b = w / W'      δ = x − m      (a + b = 1)
m' = m + b·δ     C'ᵢⱼ = a·Cᵢⱼ + a·b·δᵢδⱼ
c = W² / (W² − Q)       (1 under bias=True)
varᵢ = c·Cᵢᵢ     covᵢⱼ = c·Cᵢⱼ     corrᵢⱼ = Cᵢⱼ / √(CᵢᵢCⱼⱼ)
P = (c·(C + s·precision_prior·I))⁻¹      s = 1 on the first row, s' = a·s on each row after
partial_corrᵢⱼ = −Pᵢⱼ / √(PᵢᵢPⱼⱼ)      mahal = √(δᵀ P δ)
C_ℓ' = a·C_ℓ + a·b·(x_t − m)(x_{t−ℓ} − m)ᵀ      the co-moments' a and b, so lag 0 would be C exactly
```

The docstring's Notes, under *Bias*, give `c` in steady state at
half-lives of 20 and 2 rows.

#### Example

**It follows a correlation that moves, at the pace its half-life sets.**
This code makes two columns whose correlation jumps from 0.2 to 0.8 at row
2,000, and reads `corr_a_b` before the jump and after it:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# Two columns whose correlation jumps partway through.
rng = np.random.default_rng(1)
n = 4000
rho = np.where(np.arange(n) < 2000, 0.2, 0.8)                 # the truth: 0.2, then 0.8 from row 2000
a, e = rng.standard_normal((2, n))
pl.DataFrame({"a": a, "b": rho * a + np.sqrt(1 - rho**2) * e}).write_parquet("pair.parquet")

# Fit, and read corr_a_b around the jump.
mv = rp.spec.ew_cov("mv", features=["a", "b"], half_life=200.0, stats=["corr"])   # no clock: a half-life of 200 rows
out = pl.scan_parquet("pair.parquet").rheon.fit_predict([mv]).collect()
corr = out["mv"].struct.field("corr_a_b")
for row in (1999, 2200, 2400, 3999):                         # before the jump, one and two half-lives after, the end
    print(f"row {row}: corr {corr[row]:.2f}")
```

```text
row 1999: corr 0.20
row 2200: corr 0.49
row 2400: corr 0.68
row 3999: corr 0.80
```

One half-life after the jump, `corr` has gone half of the way to the new
level, and after two it has gone about three quarters. A shorter half-life
follows faster, and gives a noisier estimate.

#### See also

The guides to its settings:

- To flag rows that are odd only in combination, with `mahal` and
  `mahal_quantiles`, see [Flag joint outliers](how-to/flag-joint-outliers.md).
- To track principal components on a schedule, with `pca`, `pca_every` and
  `max_rows_between_pcas`, see [Track principal
  components](how-to/track-principal-components.md).
- To find which column leads another, with `lags` and `"lag_corr"`, see
  [Find which series leads](how-to/find-which-series-leads.md).

To fit, classify or screen on the same moments:

- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), a
  regression on the same running co-moments.
- [`ew_class`](#ew_class--gaussian-classification-on-ew_cov-moments) and
  [`hmm`](#hmm--which-regime-are-we-in), which classify rows by Gaussians
  on these moments.
- [`marginal`](#marginal--every-pairs-moments-kept-in-the-state), the same
  arithmetic one (feature, target) pair at a time.
- [`bank.gram()`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.gram),
  to read the moments back from the state.

#### Notes

**Each row's statistics are pandas' at the row before.** Against pandas'
`ewm(halflife=...)` mean, variance, covariance and correlation, under
either `bias`, they agree within 1e-12 on columns near zero. Under a hard
window, the means and covariances agree within 1e-12 with numpy's weighted
moments of the rows inside it, and `pc0_var` agrees with
`numpy.linalg.eigh` of the same matrix within 1e-9 relative.

**Under a `window_size`, Polars gives the same numbers on a frame in
memory, but recomputes every window** ([Forget rows past a
window](how-to/forget-rows-past-a-window.md)). Polars'
`df.rolling("t", period="3h").agg(...)`, with an exponential weight and `c`
from the weights inside the window, agrees with the spec within 1e-12. Its
work grows with `n` rows times the rows of a window, where the spec's grows
with `n`. Measured, the spec ran 24 times as fast at a 74-row
window and 1,100 times at a 4,680-row one. So use the spec for a stream, a
state to save, or speed. The docstring's Notes, under *Against Polars*,
give how to weight the rows.

### `pca` — streaming principal components

*API:* [`rheon.spec.pca`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.pca) — *Rust:* [`streampca.rs`](../crates/rheon-core/src/streampca.rs) — *Outputs:* [fields](OUTPUTS.md#pca)

Principal components of the feature columns, learned one row at a time.

Use `pca` for the leading directions of a stream's columns, the variance
along each and each row's scores, where the columns are too many for a
`k x k` covariance to be kept per group. Its state is `8 (2 + 3k + qk + q)`
bytes for `k` columns and `q` kept directions, one more than the components
written where there is room. So 2,000 columns and 12 components hold
256,120 bytes, where `ew_cov`'s covariance would hold 16 MB. For a few
hundred columns or fewer, [`ew_cov`](#ew_cov--exponentially-weighted-moments)'s
`pca` gives the exact components of the same decayed covariance, and of a
hard window too.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.pca), and
the fields it writes in [OUTPUTS.md](OUTPUTS.md#pca).

#### The update

Each method keeps the weight `W`, the sum of squared weights `Q`, the mean
`m` and the trace `tr` of the decayed covariance `C`, never `C` itself.
With `a = λW / W'`, `b = w / W'` and `d = x - m`, `C' = a C + a b d d'`.

- **`svd`** (the default), the incremental SVD that scikit-learn's
  `IncrementalPCA` runs, here a row at a time and under the decay. The `q`
  directions, each scaled by `sqrt(a)` times its singular value, and
  `sqrt(a b) d` beside them are rotated in pairs until orthogonal, and the
  `q` longest are kept. It is exact while the data's rank is at most `q`.
- **`ccipca`**, candid covariance-free incremental PCA: each direction is
  kept as its variance times its unit vector, `v' = a v + a b (u . e) u`,
  and the row deflated by each direction in turn.
- **`oja`**, Oja's rule: the directions move by `eta (V d) d'`, `eta =
  learning_rate b / tr'`, and are kept orthonormal in order.

#### When a component is written

A component whose eigenvalue equals a neighbor's is not identified: any
rotation of the two fits as well. So component `j` is written while it
passes the likelihood-ratio test of the equality of two eigenvalues
(Anderson 1963) against the component above it and the one below. The
statistic is `(n - 1) ln(((l + m)/2)^2 / (l m))` at Kish's effective size
`n`, and the test passes above 9.21, the chi-squared 99th percentile with
two degrees of freedom. No count of rows enters: the
first rows are withheld because `n` is small, and a tie is withheld for as
long as it lasts, its rows' `withheld_reason` reading `"not_separated"`.

The test reads two eigenvalue estimates, which `"svd"`'s and `"ccipca"`'s
variances are. `"oja"`'s are running means along directions that still
move. Its rate is `c/t`, with `c = learning_rate / tr` per unit of
variance. That rate turns a direction toward an eigenvector faster than
the direction's sampling error shrinks only where `c` times the gap to a
neighbor passes 1/2. Below that the variances differ by the directions'
convergence error, which the test would read as a gap. Take 20 features
with variances 4, 4 and then 1, one component and the default gain (`4 ·
3/26 = 0.46`): the tied component was written on every late row of 3 seeds
in 8. So `"oja"`
writes component `j` only where `learning_rate · gap / tr > 1/2` against
each neighbor as well, and withholds it as `"not_separated"` otherwise. A
wide stream, whose gaps are small against its trace, needs a larger
`learning_rate`.

#### Checked against

Each method is held to a third-party library where one computes it, and to
its definition where none does:

- scikit-learn's `IncrementalPCA.partial_fit`, row by row at no decay: the
  components, `explained_variance_` and `explained_variance_ratio_`, within
  7.0e-13 rad and 5.4e-13.
- scikit-learn's `PCA` and scipy's `eigh` on the decay-weighted rows, every
  direction kept: within 1.4e-14 rad and 3.3e-13.
- CCIPCA and Oja's rule against their definitions written out in numpy, to
  rounding, and against `PCA`'s subspace over ten seeds of 20,000 rows:
  within 0.0045 rad without decay and 0.0082 under a half-life of 2,000
  rows (the worst seed).

#### See also

One other model reads the same components:

- [`ew_cov`](#ew_cov--exponentially-weighted-moments)'s `pca`, the exact
  components, and of a hard window.

### `ewm_moments` — moments to any order, lagged or not

*API:* [`rheon.spec.ewm_moments`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ewm_moments) — *Rust:* [`ewm_moments.rs`](../crates/rheon-core/src/ewm_moments.rs) — *Outputs:* [fields](OUTPUTS.md#ewm_moments)

Exponentially weighted, or windowed, moments and cross-moments of columns,
to any order, each leg read at the row or lagged.

Use `ewm_moments` for the running shape of a stream's columns: a skew or a
kurtosis, a cumulant, a co-skewness of two columns, or an autocovariance or
autocorrelation at a lag, each a term you name. Each is read before its row,
so it can be a feature for that row. Every term comes from one state
whose size the terms, the lags and the window fix, whatever the stream's
length. Use [`ew_cov`](#ew_cov--exponentially-weighted-moments) for the
means, covariances and correlations of many columns and the statistics read
off them. For one term as a window column of a stream, read after its row
as the other window operators are, use the operator `rp.ops.ewm_moments`
([WINDOWS.md](WINDOWS.md)).

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ewm_moments),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#ewm_moments). The
parameters every spec takes are in [What every spec
takes](#what-every-spec-takes).

#### The update

A term names its legs and a power of each: `{"m3": {"x": 3}}` is the third
moment of `x`, `{"xy2": {"x": 1, "y": 2}}` a cross-moment, and
`{"ac1": {"x": 1, "x@1": 1}}` the first autocovariance. Row `t` holds the
weight `ω_t`, its own weight times every later row's decay, and with
`W = Σ ω_t` and `z_t` the legs' values:

```
mean           m    = Σ ω_t z_t / W
central        μ_a  = Σ ω_t (z_t − m)^a / W          (a, the powers over the legs)
raw            μ'_a = Σ ω_t z_t^a / W
standardized   s_a  = μ_a / Π_j μ_{2e_j}^(a_j/2)     (a skew, a kurtosis, a correlation)
cumulant       κ_p  = μ_p − Σ_{j=2..p−2} C(p−1, j−1) κ_j μ_{p−j}     (one leg)
```

The state keeps `W`, the means, the normalized central moments `c_a = μ_a`
below each term and the ratios `r_k = Σ ω_t^k / W^k` (`1 / r_2` is Kish's
effective number of rows). A row of weight `w` after a decay `λ` merges
into them by Pébay's rule:

```
W' = λW + w      a = λW / W'      b = w / W'      d = z − m
c'_a = Σ_{β ≤ a} C(a, β) a c_{a−β} (−b d)^β  +  b (a d)^a      (c_0 = 1, c_1 = 0)
r'_k = a^k r_k + b^k      m' = m + b d
```

Every product is at the scale of the spread, so a column at `1e6` with a
spread of 1 keeps its fourth moment, where power sums keep no digit of it,
and decay multiplies `W` alone, so a long gap moves no moment. Under
`bias=False` a term of order 2 to 4 is the form unbiased for rows drawn
independently under these weights (Rimoldini's weighted moments; scipy's
`kstat`, `skew(bias=False)` and `kurtosis(bias=False)` at equal weights).
A lag pairs a row with the row `k` learned rows back, or on a clock with
the latest at or before `t − k` within `lag_tolerance`, and the pairs are
centered on their own weighted means. Under a window the moments are read
by merging summaries of the window's parts, so a row leaving it is never
subtracted.

#### Example

**It reads a distribution's shape and a series' memory as the rows
arrive.** This code draws a gamma column, whose skew is 1 and kurtosis
4.5, and an autoregressive one whose lag-1 autocorrelation is 0.6, and
reads the last row's terms under a half-life of 10,000 rows:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# A skewed column and an autoregressive one.
rng = np.random.default_rng(5)
n = 50_000
e, ar = rng.standard_normal(n), np.zeros(n)
for t in range(1, n):
    ar[t] = 0.6 * ar[t - 1] + e[t]                            # the truth: an autocorrelation of 0.6
frame = pl.DataFrame({"g": rng.gamma(4.0, size=n), "ar": ar})    # the truth: skew 1, kurtosis 4.5

# Fit, and read the last row's terms.
model = rp.spec.ewm_moments(
    "shape",
    terms={"skew": {"g": 3}, "kurt": {"g": 4}, "ac1": {"ar": 1, "ar@1": 1}},
    kind="standardized",
    half_life=10_000.0,
)
last = rp.ModelBank([model]).fit_predict(frame)["shape"][-1]
print(f"skew: true 1.00, found {last['skew']:.2f}")
print(f"kurtosis: true 4.50, found {last['kurt']:.2f}")
print(f"lag-1 autocorrelation: true 0.60, found {last['ac1']:.2f}")
```

```text
skew: true 1.00, found 0.99
kurtosis: true 4.50, found 4.43
lag-1 autocorrelation: true 0.60, found 0.60
```

A half-life of 10,000 rows weighs the rows as about 29,000 equal ones
would (Kish's effective size, `2 · 10,000 / ln 2`), and the kurtosis, a
fourth moment, is the term furthest from its truth.

#### See also

The operator and the models beside it:

- The window operator
  [`rp.ops.ewm_moments`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.ewm_moments),
  one term as a column of a stream, each row weighing the time its value is
  held, as `rp.ops.ewm_var` weighs it.
- [`ew_cov`](#ew_cov--exponentially-weighted-moments), the means,
  covariances, correlations and principal components of many columns.
- [`audit`](#audit--what-a-streams-columns-hold), a column's lag pairs and
  tails counted once over the whole stream.

#### Notes

**Held to scipy, statsmodels and Polars.** With every weight 1 and no
decay the terms are scipy's `moment`, `skew`, `kurtosis` and `kstat`
within 1e-12, and a lagged term of order (1, 1) is numpy's
`cov(x[k:], x[:-k], bias=True)`. Under decay and unequal weights they are
numpy's weighted averages of the same products. Over a window of rows they
are Polars' `rolling_skew` and `rolling_kurtosis`.

**An autocorrelation centers each leg on its own pairs.** The pairs are centered
on their own means, so the lag-1 correlation differs from statsmodels'
`acf`, which centers on the whole stream's mean and divides by its
variance, by a few thousandths on an autoregressive series. `ew_cov`'s
`lag_corr` centers both legs on the stream's mean before the row and reads
the whole stream's variance, a third statistic: on an AR(1) series it parts
from this one by about 4e-2 at lag 2.

### `marginal` — every pair's moments, kept in the state

*API:* [`rheon.spec.marginal`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.marginal) — *Rust:* [`marginal.rs`](../crates/rheon-core/src/marginal.rs) — *Outputs:* [fields](OUTPUTS.md#marginal)

Each feature's exponentially weighted moments with each target, read back
as a table.

Use `marginal` to screen thousands of features against a few targets in one
pass. Each (feature, target) pair is its own two-column
[`ew_cov`](#ew_cov--exponentially-weighted-moments), so for `p` features
and `T` targets a row takes O(p·T) time, where one `ew_cov` over all the
columns would take O((p + T)²). Use `ew_cov` for the joint covariance of a
few columns, and
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) or
[`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs) for a joint fit
of the target on every feature.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.marginal).
Each row writes only `weight_sum` ([OUTPUTS.md](OUTPUTS.md#marginal)), and
`bank.marginal()` reads the pairs from the state as a table after the run.
[Its
docstring](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.marginal)
defines each column. The parameters every spec takes are in [What every spec
takes](#what-every-spec-takes).

Five settings extend the pairs, each off by default, and the first two add
columns to the table:

| setting | what it adds |
|---|---|
| `lags`, with `serial_rule` | the pair's moments at each lag: a count that allows for rows that resemble their neighbors, and lead/lag correlations |
| `bins` | the target's moments inside each bin of a feature: the relations a correlation cannot see |
| `window_size` | the pairs over a recent stretch |
| `feature_moments="shared"` | speed: one feature moment for every target, and the same table where every target is on every row |
| `shards` | speed: one wide spec on every thread, and the same numbers at any count |

When a group closes, `bank.closed_groups()` writes every pair as `pair_*`
columns: a list over the pairs, such as `pair_split_gain`, or a list of
lists for the columns that hold a list per pair, the `lag_corr_*` and
`bin_*` columns, such as `pair_lag_corr_xx` and `pair_bin_n`.

#### The update

Each pair (feature `j`, target `t`) keeps its own `m_x`, `S_xx` and `S_xy`,
over its target's rows, beside the target's `m_y` and `S_yy`. Per target
`t`, on a row where `y_t` is present, with `W_t` the weight behind that
target before the row and `Q_t` the sum of its squared weights:

```
W'_t = λW_t + w        a = λW_t / W'_t        b = w / W'_t        Q'_t = λ²Q_t + w²
S'_yy = a·S_yy + a·b·(y_t − m_y)²             S'_xx = a·S_xx + a·b·(x_j − m_x)²
S'_xy = a·S_xy + a·b·(x_j − m_x)(y_t − m_y)    m' = m + b·(value − m)      for m_y and each m_x
n_kish = W_t² / Q_t
```

`value` is the row's `y_t` for `m_y` and its `x_j` for `m_x`, and every
deviation is from the means before the row. A null target decays that
target (`W_t` by `λ`, `Q_t` by `λ²`) and learns nothing for it, and a null
feature skips the row.

For unit weights a step `Δclock` apart, `n_kish` tends to
`(1 + λ)/(1 − λ)`, where `λ` is that step's decay. Under
`bin_stat="huber"`, each bin keeps a location `m` and `Ω`, the weight its
rows count for. With `c` the cut, `bin_huber_delta` (default 1.345) times
the target's robust scale `σ_t`, both read before the row, and `e = y − m`
the row's signed deviation from the bin's location:

```
ω = w·min(1, c/|e|)      Ω' = λΩ + ω      m' = m + (ω/Ω')·e
```

The location is the mean over the weights `ω`, so its fixed point is the
Huber M-location of the bin's rows. One row moves it by at most
`c·w/Ω' = c·w/(λΩ + ω)`, which under decay is up to `1/λ` times `c·w/Ω`.
The docstring's update gives the form `feature_moments="shared"` takes.

#### Example

**A screen finds the few features that matter among many.** This code makes
500 features, of which five move the target, screens every pair in one pass,
and prints the six with the largest `|t_stat|`:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# Many features, of which only a few move the target.
rng = np.random.default_rng(2)
n, p = 2000, 500
x = rng.standard_normal((n, p))
y = x[:, [7, 42, 99, 250, 401]] @ np.array([0.5, -0.4, 0.3, 0.3, -0.2]) + rng.standard_normal(n)   # five matter
features = [f"f{j}" for j in range(p)]
pl.DataFrame(x, schema=features).with_columns(y=y).write_parquet("screen.parquet")

# Screen every pair in one pass, then rank by |t_stat|.
screen = rp.spec.marginal("screen", targets=["y"], features=features, half_life=float("inf"))   # every row weighs the same
bank = rp.ModelBank([screen])
bank.fit(pl.scan_parquet("screen.parquet"))                  # learn, and keep only the state
table = bank.marginal("screen").sort(pl.col("t_stat").abs(), descending=True)
for r in table.head(6).iter_rows(named=True):
    print(f"{r['feature']}: corr {r['corr']:+.2f}, t_stat {r['t_stat']:+.1f}")
```

```text
f7: corr +0.40, t_stat +19.2
f42: corr -0.34, t_stat -16.0
f250: corr +0.23, t_stat +10.7
f99: corr +0.23, t_stat +10.6
f401: corr -0.19, t_stat -8.5
f275: corr -0.07, t_stat -3.1
```

The five planted features lead, in the order of their coefficients, and the
best of the 495 others trails at a `t_stat` of −3.1. With 495 chances, a
value that size is what noise alone gives, which is why `t_stat` ranks
pairs and is no p-value.

#### See also

The guides to its settings:

- To rank the pairs by `t_stat`, correct the ranking with `lags` and
  `serial_rule`, and read `split_gain` and `bin_stat="huber"`'s bins, see
  [Screen features against a target](how-to/screen-features-against-a-target.md).
- [Forget rows past a window](how-to/forget-rows-past-a-window.md) shows how
  to give a screen a `window_size` when a relation may flip sign.
- [Split one wide spec across
  threads](how-to/use-more-cores.md#split-one-wide-spec-across-threads)
  shows how to spread one wide spec over the threads with `shards`.

To read the pairs, or to fit the features jointly:

- [`bank.marginal()`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.marginal),
  the table of pairs and every column it holds.
- [`ew_cov`](#ew_cov--exponentially-weighted-moments), the joint
  covariance whose arithmetic each pair uses.
- [`huber`](#huber--robust-regression), whose weighting `bin_stat="huber"`
  gives each bin.
- [`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs), to choose
  the features of a joint fit.

#### Notes

**With several targets present on every row, give `feature_moments="shared"`
for speed.** Each feature then keeps one mean and variance, over every
learned row, in place of one per pair over its target's rows. At 20,000
pairs that runs 2.7 times as fast at ten targets, with a state under half
the size and the same table to the bit (measured on 2026-09-29 on the
model alone over 4,000 rows, the best of five runs of
`crates/rheon-core/examples/marg_bench.rs`).
Where a target is absent on some rows, `var_x` and the center of `cov` then
come from every row. That is a different estimator, sound only where the
absence says nothing about the feature, so otherwise keep the default. It
takes `lags`, and refuses a `window_size`.

**Under `"huber"`, a bin's first row is judged against `m_t`, the target's
own location, the only location there is,** and winsorized at `m_t ± c`. A
row far from the bin still counts, at weight `c/|e|` of a row. A spike among
a target's first two rows, before it has a scale, counts in full, and leaves
the bins it lands in as a mean's start does.

**The first rows are refit once, `bin_refit_rows`, because one pass can be
captured.** Against statsmodels' M-location of each bin over the rows so far
(16 bins, spikes of about 20 of the target's scales, no decay), the worst
bin of the worst of 50 streams, in the target's scales:

| spikes | after 2,000 rows: one pass / refit | after 20,000 rows: one pass / refit |
|---|---|---|
| 5% | 0.36 / 0.005 | 0.03 / 0.001 |
| 12% | 1.29 / 0.015 | 0.23 / 0.004 |
| 20% | 3.5 / 0.14 | 1.0 / 0.057 |

**The one pass alone lands near the converged estimate of all the rows at
once, and the refit brings it closer.** The reference is statsmodels'
M-location of each bin (`estimate_location` with `HuberT(1.345)` at the
target's MAD), on 20,000 rows in 16 bins with no decay, with spikes of 50
on a share of the rows. Against it, the largest gap over ten streams of the
one pass alone (`bin_refit_rows=0`), in units of the target's scale:

| spikes | `"huber"`, one pass alone, against the converged estimate | the mean against it |
|---|---|---|
| none | 0.0006 | 0 |
| 1% | 0.0010 | 0.45 |
| 5% | 0.0016 | 1.51 |
| 12% | 0.030 | 2.95 |

With the refit of the first 500 rows, the worst of 50 streams, a harder test than the table's ten, was
7.7e-16, 3.0e-4, 1.2e-3 and 3.1e-3 of the scale at the same four shares.
The one pass's gap comes from its first rows, which are weighed against a
location and a scale that are still moving. On 2,000 rows, 125 to a bin, it was 0.013 at
1% and 0.37 at 5%, the largest from a spike on a stream's second row. The
location steps over `Ω`, the weight its rows count for, rather than `W`, the
weight they carry. From a spike `E₀` off at full weight `Ω₀`, while each row
is clipped the error decays as `E(n) = E₀·exp(−c·n/(Ω₀E₀))`. So it is within
one scale `σ` after about `(Ω₀E₀/c)·ln(E₀/σ)` rows. Simulated with no
decay, unit weights and `σ` fixed, a spike of 10, 30 or 100 scales was
within one scale after 27, 101 or 381 rows. Over `W` a row moves the
location back at most `c·w/W`, so `n` rows recover at most `c·ln n`, 13
scales in 20,000 rows. In the same simulation the spike of 30 scales was
still 17 scales off after 20,000 rows, and the one of 100 still 87.

Measured at 16 bins, `"huber"` ran at 0.75 to 0.9 times the mean's
throughput at 100 and 10,000 features and one and ten targets. The
saved state was a fifth to a third larger.

**Under `serial_rule="geometric"`, each series' autocorrelations are
fitted as `ρ(l) = φˡ`, and the bracket's sum is taken in closed form,
`2φ_xφ_y/(1 − φ_xφ_y)`,** so the kept lags need not be dense. `"truncated"`
suits dense lags, and `"bartlett"`, which weights lag `l` by
`1 − l/(L + 1)` for `L` the longest kept lag, suits dense lags whose long
ones are noisy.

**Under a hard window, the lag moments are kept about each series' first
value, which truncates exactly, and read about the hard window's means when
asked.** A correlation centered at the running mean instead carries the
history's mean into the hard window. On a hard window whose rows had moved
to another level, it read a lag correlation of 22, outside `[−1, 1]`.

**Each pair matches a two-column `ew_cov`.** Its means and variances equal those an
`ew_cov(bias=True)` over the two columns reports, to the bit. Its `cov` and
`corr` agree within a few roundings: the pair rounds each row's product as
`(a·b·dx)·dy`, while `ew_cov`, which keeps one triangle of its symmetric
matrix, rounds it as `(a·b·dy)·dx`. A pair's `corr` agrees with pandas'
`ewm(halflife=...).corr` within 1e-12 relative, and under a hard window its
moments agree with numpy's weighted moments of the rows inside it. The
docstring's Notes give the serial count's check on AR(1) series, the Huber
locations' against statsmodels, and the lags' under a hard window against
statsmodels' `acf`.

### `deco` — one correlation for the whole matrix

*API:* [`rheon.spec.deco`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.deco) — *Rust:* [`deco.rs`](../crates/rheon-core/src/deco.rs) — *Outputs:* [fields](OUTPUTS.md#deco)

Engle and Kelly's dynamic equicorrelation, one number for the whole
correlation matrix of the features.

Use `deco` (Engle & Kelly) for one number that moves with the
correlation of many series. It replaces the `m(m−1)/2` free entries of the
correlation matrix of `m` series with their average, the
*equicorrelation*, and estimates that in O(m) a row, where keeping every
entry moving takes O(m²). Use
[`ew_cov`](#ew_cov--exponentially-weighted-moments) for each pair's
correlation, and `deco`'s own `blocks` for one number per block of features
and one per pair of blocks.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.deco),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#deco). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes).

#### The update

The row is standardized against the means and variances as they stood
before it, `ζ_i = (x_i − m_i)/√v_i`. The sums `S₁ = Σ ζ_i` and
`S₂ = Σ ζ_i²` run over the `n` features that have a standardized value.
The row's estimate `u` is Engle and Kelly's eq. 20, the mean product of two
standardized values over the mean square, which lies in `[−1/(n − 1), 1]`
(their Lemma 2.3). Between two blocks `A` and `B` the estimate is `u_AB`:

```
u    = A / V,       A = (S₁² − S₂) / (n(n − 1)),   V = S₂ / n
u_AB = C / √(V_A·V_B),   C = S₁_A·S₁_B / (n_A·n_B)
```

`u` is a ratio over one row, and the mean of a ratio is not the ratio of
the means. It reads toward 0, at 1/3 for a true 0.6 with two series and
about 0.41 with many. Neither more series nor a longer half-life shrinks
that, since every row sees one draw of the common factor. So the
reported level, `rho`, averages the numerator and the denominator and
divides once, with `α` and `β` the `"linear"` dynamic's two weights:

```
W' = λW + w,  b = w/W',  Ā' = Ā + b·(A − Ā),  V̄' = V̄ + b·(V − V̄)
"ew":      ρ = Ā / V̄
"linear":  N' = (1 − α − β)·Ā' + α·A + β·N,   D' = (1 − α − β)·V̄' + α·V + β·D,   ρ = N / D
```

The `"ew"` level is the weighted Gaussian maximum-likelihood estimate of
the equicorrelation over the exponentially weighted window: their Lemma
2.4, which reads `u` as the one-row estimate, with the sums taken over the
window. The `"linear"` level is a DCC model's correlation recursion,
targeted at the window's means and averaged over its off-diagonal and its
diagonal. The level learns from a row only once the standardizer's weights
hold more than five effective rows, where a standardized row's fourth
moment exists. The docstring's Notes give the rest. *The clock* says how
the `"linear"` level treats a capped gap and a row's weight, *The
standardizer's warm-up* what the warm-up reads, and *Departures from the
paper* why the intercept is `(1 - alpha - beta)` times the means.

#### Example

**`rho` moves with the average correlation, and reads it.** This code
makes ten series whose every pair correlates at 0.2 and then, from row
2,000, at 0.6. It runs `deco` beside an `ew_cov` over the same columns, and
prints `rho` beside the mean of the 45 pairwise correlations:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# Ten series whose common correlation rises partway through.
rng = np.random.default_rng(3)
n, m = 4000, 10
rho = np.where(np.arange(n) < 2000, 0.2, 0.6)                 # the truth: every pair at 0.2, then 0.6 from row 2000
common, own = rng.standard_normal(n), rng.standard_normal((m, n))
names = [f"s{i}" for i in range(m)]
pl.DataFrame(dict(zip(names, np.sqrt(rho) * common + np.sqrt(1 - rho) * own))).write_parquet("ten.parquet")

# deco beside an ew_cov over the same columns.
eq = rp.spec.deco("eq", features=names, half_life=100.0)
pairs = rp.spec.ew_cov("pairs", features=names, half_life=100.0, stats=["corr"])
out = pl.scan_parquet("ten.parquet").rheon.fit_predict([eq, pairs]).collect()
# rho beside the mean pairwise correlation, before and after the rise.
level = out["eq"].struct.field("rho")
mean_corr = out["pairs"].struct.unnest().select(pl.mean_horizontal(pl.selectors.starts_with("corr_"))).to_series()
for row in (1999, 3999):
    print(f"row {row}: truth {rho[row]:.1f}, deco's rho {level[row]:.2f}, the mean of ew_cov's corr {mean_corr[row]:.2f}")
```

```text
row 1999: truth 0.2, deco's rho 0.19, the mean of ew_cov's corr 0.19
row 3999: truth 0.6, deco's rho 0.59, the mean of ew_cov's corr 0.60
```

`deco` takes O(m) a row where the `ew_cov` takes O(m²), and reads the same
average.

#### See also

The guides to its settings:

- To read `u` and `rho` as a signal, per block with `blocks`, and to leave a
  constant column out of `features`, see [Track the average
  correlation](how-to/track-the-average-correlation.md).

For each pair's correlation, or a test of a change in it:

- [`ew_cov`](#ew_cov--exponentially-weighted-moments), every pair's
  correlation, at O(m²) a row.
- [`corrchange`](#corrchange--has-the-correlation-structure-changed), a test
  for a change in the correlation structure.

#### Notes

**`u` is biased toward 0, as Engle and Kelly note, and `rho` is not.**
`u` reads 0.130 for a true 0.2 and 0.379 for a true 0.6 on five series,
where `rho` reads 0.198 and 0.607 at a half-life of 100:
[`deco`: an equicorrelation that moves](REGIMES.md#7-deco-an-equicorrelation-that-moves)
measures both.

**The `"ew"` level is held to scipy's weighted Gaussian likelihood,
maximized over the window, and the `"linear"` level to the full `n × n`
DCC recursion averaged.** The `"linear"` level at `alpha` and `beta` of 0 is
the `"ew"` level, to the bit, and `loglik` is held to the dense Gaussian
density of the correlation matrix the levels describe. The docstring's
Notes, under *Departures from the paper*, give the paper's alternative row
estimate, which this model does not compute.

### `rcov` — a block's realized covariance, robust to noise

*API:* [`rheon.spec.rcov`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.rcov) — *Rust:* [`rcov.rs`](../crates/rheon-core/src/rcov.rs) — *Outputs:* [fields](OUTPUTS.md#rcov)

A block's realized covariance of the features, robust to microstructure
noise.

Use `rcov` for the covariance of returns over each block of a tick stream,
such as a trading day, where noise in each price biases a plain realized
covariance. For series not observed together, use its `"preavg"` kind. Use [`ew_cov`](#ew_cov--exponentially-weighted-moments)
for running moments that decay and are read on every row.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.rcov).
It requires `group`, naming the block column, and `group_close`, and
refuses a `half_life` or `lam`, since a block's close ends its memory. Each
row writes only `weight_sum` ([OUTPUTS.md](OUTPUTS.md#rcov)), and each
block's estimate is that block's row of `bank.closed_groups()` when it
closes. The parameters every spec takes are in [What every spec
takes](#what-every-spec-takes).

#### The update

Each row's returns `x` enter the block's sums over lags, and at the close
the estimator `kind` names reads them. In the kernel, `Gamma_h` sums the
products of returns `h` rows apart, up to a bandwidth `H`, with `k` the
Parzen weight. `xi² = omega2 / iv_sparse` is the noise variance over the
sparse integrated variance. `"preavg"` pre-averages the returns over `k_n`
rows with `g`, its window scaled by `theta` (default 1), and `rcov_n`
counts the pre-averaged terms. `psi1` and `psi2` are its finite-sample
constants (1 and 1/12 in the limit). Over a block of `n` returns, with
`block_rows` the block's expected length:

```
"plain":    sum_j x_j x_jᵀ
"kernel":   K = sum_{h=−H}^{H} k(h/(H+1))·Gamma_h      Gamma_h = sum_j x_j x_{j−h}ᵀ,   Gamma_{−h} = Gamma_hᵀ
            H = ceil(c*·xi^(4/5)·n^(3/5)),  c* = 3.5134, unless bandwidth= fixes it
"preavg":   Ybar_i = sum_{j=1}^{k_n−1} g(j/k_n)·x_{i+j}      g(x) = min(x, 1 − x)
            MRC = n / rcov_n / (psi2·k_n) · sum_i Ybar_i Ybar_iᵀ
            psd=True:   MRC                                  k_n = ceil(theta * block_rows^0.6)
            psd=False:  (MRC − b·sum_j x_j x_jᵀ) / (1 − b)    k_n = floor(theta * sqrt(block_rows)),  b = psi1 / (2 psi2 k_n^2) → 6 / k_n^2
```

**The estimate reads no future row.** The kernel's jittered end points
average observations already in the state at close, and a product enters
`Gamma_h` only once both its legs are final.

#### Example

**The kernel recovers a correlation that noise hides from the plain sum.**
This code makes 20 blocks of two prices whose returns correlate at 0.6, each
tick measured with noise, and prints each estimator's mean correlation over
the blocks:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# Blocks of two noisy prices whose underlying returns correlate.
rng = np.random.default_rng(4)
blocks, rows = 20, 1000
z = rng.standard_normal((2, blocks * rows))
latent = np.cumsum([z[0], 0.6 * z[0] + 0.8 * z[1]], axis=1)  # two efficient prices whose returns correlate at 0.6
observed = latent + rng.normal(0.0, 0.7, latent.shape)        # each tick measured with noise
pl.DataFrame({
    "block": np.repeat(np.arange(blocks), rows),              # a block id: a trading day, say
    "p1": observed[0], "p2": observed[1],
}).write_parquet("blocks.parquet")

# Each price's returns within its block, fed to both estimators.
returns = pl.scan_parquet("blocks.parquet").with_columns(pl.col("p1", "p2").diff().over("block"))   # within each block
specs = [
    rp.spec.rcov(kind, features=["p1", "p2"], group="block", group_close="monotone", kind=kind, block_rows=rows)
    for kind in ("plain", "kernel")
]
bank = rp.ModelBank(specs)
bank.fit(returns)
# Each estimator's mean correlation over the blocks that closed.
for kind in ("plain", "kernel"):
    closed = bank.closed_groups(kind)                        # one row per block that closed
    corr = closed["rcorr"].list.get(1)                         # the upper triangle, row by row: [1, r, 1]
    print(f"{kind}: mean correlation {corr.mean():.2f} over {closed.height} blocks (truth 0.60)")
```

```text
plain: mean correlation 0.29 over 19 blocks (truth 0.60)
kernel: mean correlation 0.59 over 19 blocks (truth 0.60)
```

The last block never closes, because no later block arrives to close it
under `group_close="monotone"`, so 19 of the 20 report.

#### See also

The guides to its settings:

- To choose `kind` for the returns you have, and to read each block's estimate, see
  [Estimate a block's covariance from
  ticks](how-to/estimate-a-blocks-covariance-from-ticks.md).

To put the prices on one grid before taking returns, or to read each block's estimate:

- [`ew_cov`](#ew_cov--exponentially-weighted-moments), running moments that
  decay, read before each row.
- [Align series that tick at their own
  times](how-to/align-series-that-tick-at-their-own-times.md), to put the
  prices of several series on one grid with `rp.stream.refresh_time` before
  taking returns.
- [`bank.closed_groups()`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.closed_groups),
  which returns each block's estimate when the block closes.

#### Notes

**`"plain"` equals `n` times an `ew_cov(lam=1)`'s uncentered second moment
at close, to the bit.** It is the cross-check, and the reference the other
two are measured against. The jittered kernel agrees with hfhd's `krvm` on
the jittered prices within 1e-13 of the estimate's largest entry, over
three features, cross terms included. `"preavg"` in its default
form, `psd=True`, agrees with hfhd's `mrc(bias_correction=False)` within 1e-13
of the estimate's largest entry, after two mappings, with `block_rows` set to match. The test
gives hfhd's prices a leading copy of the first, since hfhd's
pre-averaging starts one return later. hfhd then counts `n + 1` returns,
`n` the block's, so the test sets `block_rows` to `n + 1`, and both take
the same `k_n`. The test
then replaces hfhd's scale `(n + 1)/(n − k_n + 3)` with this library's
`n/(n − k_n + 2)`. [`rcov`: three estimators against each block's
truth](REGIMES.md#8-rcov-three-estimators-against-each-blocks-truth) measures
each kind against each block's truth.

### `audit` — what a stream's columns hold

*API:* [`rheon.spec.audit`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.audit) — *Rust:* [`audit.rs`](../crates/rheon-core/src/audit.rs) — *Outputs:* [fields](OUTPUTS.md#audit)

A count of what a stream's columns hold, read in one pass by a spec that
fits nothing and predicts nothing.

Use `audit` before choosing a model, to find out what is wrong with the
data, such as a sentinel value, a frozen feed or a heavy tail. Run it alone,
or beside the models in the same pass, and
[`bank.check()`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.check)
reads its counts and says what they mean. Use `bank.describe()` for a few
statistics of the columns a model has already been fed.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.audit).
Each row writes only `weight_sum` ([OUTPUTS.md](OUTPUTS.md#audit)), and
`bank.audit()` reads the counts from the state as a table after the run.
[Its
docstring](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.audit)
defines each column. The parameters every spec takes are in [What every spec
takes](#what-every-spec-takes).

#### The update

Per column, each row's value `x` is first classified. A null, a NaN,
`+inf`, `-inf` and a finite value past `1e100` in magnitude each add one to
a count of their own, and anything else is usable. A usable value enters
the moments by Terriberry's update of the central sums `M2`, `M3` and `M4`,
with `n` the usable values before it. Its value's counter adds one, and
past `distinct_cap` distinct values the counters run as a Misra-Gries
summary. Nothing decays. A row of weight 0 reaches no count, moment,
counter or pair, nor the clock's steps, as it counts in no clock of any
model; a bank's `audit` takes no weight column, so every row it reads
weighs 1 and counts:

```
n' = n + 1      d = x − mean      e = d / n'      t = d·e·n
mean' = mean + e
M4' = M4 + t·e²·(n'² − 3n' + 3) + 6e²·M2 − 4e·M3
M3' = M3 + t·e·(n' − 2) − 3e·M2
M2' = M2 + t
```

#### Example

**An audit finds a frozen feed and a sentinel value.** This code makes a
price that stops moving at row 1,200 and a size column that holds `-999` on
one row in 40. It prints what `bank.audit()` measured, and what
`bank.check()` reads from it:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# A feed with two faults planted: a frozen price and a sentinel size.
rng = np.random.default_rng(5)
n = 2000
price = 100 + np.cumsum(rng.normal(0.0, 0.1, n))
price[1200:] = price[1199]                                    # the feed froze at row 1200
size = rng.exponential(1.0, n)
size[::40] = -999.0                                           # a sentinel standing for "missing", on one row in 40
pl.DataFrame({"price": price, "size": size, "flow": rng.standard_normal(n)}).write_parquet("feed.parquet")

# Audit the three columns, then print the counts and what they mean.
bank = rp.ModelBank([rp.spec.audit("audit", columns=["price", "size", "flow"])])
bank.fit(pl.scan_parquet("feed.parquet"))
for r in bank.audit().iter_rows(named=True):                  # what each column holds
    print(f"{r['column']:>5}: longest run of one value {r['longest_run']}, smallest value {r['min']:g}")
for r in bank.check().iter_rows(named=True):                  # what that means, the worst first
    print(f"{r['severity']}: {r['code']} on {r['column']}")
```

```text
price: longest run of one value 801, smallest value 97.3958
 size: longest run of one value 1, smallest value -999
 flow: longest run of one value 1, smallest value -3.22357
warning: frozen on price
warning: random_walk on price
warning: sentinel on size
info: heavy_tails on size
```

Each finding's `message` says what to do, and links its section of
[DATA-QUALITY.md](DATA-QUALITY.md), which shows what the problem does to
each model and how to fix it with Polars.

#### See also

To judge the counts, and to fix what they find:

- [`bank.check()`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.check),
  what the counts say is wrong.
- [DATA-QUALITY.md](DATA-QUALITY.md), each finding, what it does to each
  model and how to fix it with Polars.
- [Find a duplicated
  column](how-to/check-data-quality.md#find-a-duplicated-column), with
  `pairs=True`, and [read every group as one
  stream](how-to/check-data-quality.md#read-every-group-as-one-stream),
  with `bank.audit(pooled=True)`.

#### Notes

**Its statistics are the libraries' own, from the raw values.** The mean
agrees with numpy's within 1e-14 relative, or 1e-12 where it is near 0, and the standard deviation with
`np.std(ddof=1)` within 1e-10. The skew and excess kurtosis agree with
scipy's biased estimators within 1e-8 relative, and so does `unit_root_t`
with statsmodels' `adfuller`, with a constant and no lags. `autocorr` agrees with numpy's
correlation of a column with its own lag, and the median and median
absolute deviation are numpy's while the counts are exact. Past the cap,
they are within about 1% of the rows in rank. Each pair's correlation
agrees with `np.corrcoef` over the rows where both columns are usable,
within 1e-13 relative, in each group and in the pooled merge.

### `spectral` — how strong is each cycle?

*API:* [`rheon.spec.spectral`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.spectral) — *Rust:* [`spectral.rs`](../crates/rheon-core/src/spectral.rs) — *Outputs:* [fields](OUTPUTS.md#spectral)

The power of each target at chosen periods, from an exponentially weighted
Fourier transform on the clock, read as scipy's periodogram.

Use `spectral` to watch how strong a known cycle is as the stream runs: a
daily rhythm in a load, a weekly one in sales, a line in a vibration. Each
row reports the power at each period before the row, under the spec's
`half_life`. Its parameters, their defaults and what each refuses are in
its [docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.spectral),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#spectral). It takes no
features.

#### The update

Per row of clock step `Δ`, weight `w` and target `x`, with the rotation
`r = e^{2πi f Δ}` for each frequency `f`, the inverse of a period:

```
X ← λ·r·X + w·x        U ← λ·r·U + w
A ← λ·A + w            B ← λ·B + w·x        Q ← λ²·Q + w²
P = c·|X − m·U|²·spacing / Q      (density; m = B / A, the weighted mean)
```

`X` is the transform of the rows at amplitudes `w·λ^age`, referred to the
current clock, and `U` the amplitudes' own, which removes the mean. `c` is
2, and 1 at the zero frequency and the Nyquist. Under `scaling="spectrum"`
the power is `c·|X − m·U|² / A²`. The `sin` and `cos` come from one
library on every platform, so the state is the same bits everywhere. A
session change or a gap past `gap_cap` empties the transform, since the
phase across it is not known.

**Lomb-Scargle.** Under `scaling="lomb_scargle"` the power at each period
is the generalized Lomb-Scargle power: the share of the target's weighted
variance that a sinusoid of that period explains, between 0 and 1. It
comes from the weighted least squares fit of `a·cos + b·sin`, and a
constant with `detrend`. It needs no even clock and has no Nyquist limit, so it is the
read for an uneven clock. It keeps two sums more: the amplitudes'
transform at twice each frequency, `V ← λ·r²·V + w`, and `S ← λ·S + w·x²`.
The fit's cross-products are read from them: `CC = (A + Re V)/2`,
`SS = (A − Re V)/2`, `CS = Im V / 2`. Where the rows' phases have not yet
spread over enough of a cycle to tell a cosine from a sine, the power is
null. It equals scipy's `lombscargle(normalize=True, weights=a,
floating_mean=detrend)` of the rows to 1e-9.

**A window.** Under `window_size` the power is of the rows less than
`window_size` old alone, the sliding-window transform: with no decay,
scipy's `periodogram` of the window's rows. The rows are kept in blocks,
closed on `snapshot_every` and `max_rows_between_snapshots` (every row by
default), each block's sums referred to its own origin, the clock of its
first row. A closed block is turned once to a reference origin and added
to a running total; it leaves whole once its first row is past the window.
The reference moves once it is two windows old and the total is written
afresh, so no sum is turned row after row. Over ten million rows, with
the clock starting near 0 or near 2^40, the power is within 1e-12 of the
direct sum over the window's rows. `closed` and `window_budget` are as
`ewridge`'s.

#### Example

**A daily cycle found in an hourly series:**

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# A cycle of 24 rows under noise of the same size.
rng = np.random.default_rng(1)
t = np.arange(24 * 60, dtype=float)
y = np.sin(2 * np.pi * t / 24) + rng.standard_normal(t.size)
pl.DataFrame({"y": y}).write_parquet("hourly.parquet")

spec = rp.spec.spectral("s", targets=["y"], periods=[24.0, 10.0], half_life=float("inf"))
out = pl.scan_parquet("hourly.parquet").rheon.fit_predict([spec]).collect()
last = out["s"][-1]
print(f"24 h: {last['power_y_p0']:.1f}, 10 h: {last['power_y_p1']:.1f}")
```

```text
24 h: 684.0, 10 h: 1.4
```

The daily period's power is hundreds of times the other's.

#### See also

When the question is a forecast or a spread rather than a cycle:

- [`ar`](#ar--an-autoregression-on-the-targets-own-past), to forecast a
  series from its own past.
- [`ew_cov`](#ew_cov--exponentially-weighted-moments), for a series'
  variance rather than its cycles.

#### Notes

**With no decay it is scipy's `periodogram`.** On an even clock of spacing
0.5, the power before a row equals `periodogram(x − m, fs=2, window=a,
detrend=False)` of the rows before it to 1e-9, `a` the decay's amplitudes.
That holds at two frequencies and the Nyquist, in both scalings, with no
decay and under a half-life. On an uneven clock with weights, where scipy has no
periodogram, it equals the definition written out with numpy to 1e-9.

### `garch` — how large will the next return be?

*API:* [`rheon.spec.garch`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.garch) — *Rust:* [`garch.rs`](../crates/rheon-core/src/garch.rs) — *Outputs:* [fields](OUTPUTS.md#garch)

The GARCH family: the variance of each row's return given the rows before
it, with the process's parameters learned one row at a time.

Use `garch` on returns, or on any zero-mean residual whose size clusters in
time, when you want each row's variance forecast before the row arrives: a
risk limit, a position size, a standardized residual for a detector
downstream. Each target is its own series with its own parameters. The
start's rows (2,000 by default) report nothing; after them every row
reports its variance, its standardized residual and, at each of `forecast_rows`,
the variance further ahead and the variance of the return over the rows to
it.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.garch),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#garch). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes),
apart from `half_life` and `lam`, which it refuses: the recursion is per
observation and the estimate weighs every row it has learned. Its
`features`, if any, are inputs of the variance equation (GARCH-X, below). A row's weight scales its evidence. A row of weight 0, or whose
return is null, steps nothing: the next row's lags are those of the last row
learned, as if it were not there; it is still read.

#### The processes

With `s_t` the variance of row `t`'s return `e_t` given the rows before it,
and `z_t = e_t / √s_t`:

```
garch    s_t = ω + α e²_{t−1} + β s_{t−1}
gjr      s_t = ω + α e²_{t−1} + γ 1[e_{t−1} < 0] e²_{t−1} + β s_{t−1}
egarch   ln s_t = ω + α (|z_{t−1}| − √(2/π)) + γ z_{t−1} + β ln s_{t−1}
figarch  s_t = ω/(1 − β) + Σ_{j=1..M} λ_j e²_{t−j}
         λ_1 = φ − β + d,  λ_j = β λ_{j−1} + δ_j − φ δ_{j−1},  δ_j = d Π_{i=2..j} (i − 1 − d)/i
```

The process is the `vol` parameter. GARCH(1,1) is the default. In GJR, a fall
lifts the variance more than a rise of the same size. EGARCH runs in
the logarithm of the variance, so no sign constraint keeps it positive.
FIGARCH's memory of the squared returns decays like a power of the lag
rather than geometrically, cut at `truncation` lags (1,000). `dist` picks the
innovations: normal; Student's t, whose degrees of freedom `ν` are learned;
Hansen's skewed t, which learns a skew `λ` too. The t takes `garch` and
`gjr`, the skewed t `garch` alone. Every lag before the first row is the
backcast, the `0.94^i`-weighted mean of the first 75 squared returns.

**Inputs (GARCH-X).** `features` are inputs `x` of the variance equation,
each with a coefficient `π` of its own, learned beside the process's:

```
garch-X   s_t = ω + π'x_t + α e²_{t−1} + β s_{t−1}                       π ≥ 0, x_t ≥ 0
gjr-X     s_t = ω + π'x_t + α e²_{t−1} + γ 1[e_{t−1} < 0] e²_{t−1} + β s_{t−1}
egarch-X  ln s_t = ω + π'x_t + α (|z_{t−1}| − √(2/π)) + γ z_{t−1} + β ln s_{t−1}
```

The row's own `x_t` is read, as every model reads its features: they are
known before the row's target. So lag a column that measures the day
before, such as a realized variance, to make it known before the return
it explains; `rugarch`'s `external.regressors` read the same row. Under
GARCH and GJR each `π` is at least 0 and an input must be at least 0, so
the variance stays positive. A row whose input is below 0 reports nothing,
`withheld_reason` `input_negative`, and is learned as a row of weight 0,
as a row with a null feature is. Under EGARCH the input enters the
log-variance, and `π` and the input take any sign. Each `π`'s sensitivity
is the filter's own recursion driven by the input, `∂s_t/∂π = x_t + β
∂s_{t−1}/∂π` (EGARCH's carries `β − (α |z| + γ z)/2` in place of `β`).
FIGARCH and variance targeting take no inputs; at most 8.

#### The update

Each learned row of weight `w_t` steps the parameters by the score of its
log-likelihood `l_t = log f(z_t) − ½ ln s_t`. The score is `g_t = ½ w ∂ln
s_t/∂θ`, with `w = z² − 1` under the normal and a bounded `w` under a t, and
the sensitivity `∂ln s_t/∂θ` is carried by its own recursion:

```
a_t = w_t / W_t          W_t the weight learned, the start's included
H  ← H + a_t (I_t − H)   I_t = ½ ∂ln s ∂ln sᵀ, or the score's outer product under a t
θ  ← Proj_H(θ + a_t H⁻¹ g_t)
```

`Proj_H` projects onto the process's admissible set, the persistence at most
`1 − 10⁻⁴`, in `H`'s own metric. A Newton step projected in the Euclidean
metric is no ascent where the bounds hold parameters the likelihood barely
tells apart, such as `α + β` near 1; in `H`'s metric it is. A t's `ν` is
stepped as `ln(ν − 2)`, so no step crosses 2.

**EGARCH's guard keeps the filter that runs contracting.** Its set holds
`α ≥ |γ|` and `α ≤ 1`; its variance stays within a factor of `10⁶` of the
0.94-weighted mean square, as `arch`'s does; and the mean of `ln |β − (α |z|
+ γ z)/2|` over the standardized residuals the filter has met, the
recursion's Lyapunov exponent, is held below 0 at the parameters as they
stand, a step that would break it cut back along the step to where it
holds. While it holds, the filter forgets its start and a small error in it
dies out. It agrees with `arch` wherever `arch`'s own filter contracts, and
carries no proof that the estimate it guards is consistent. The guard with
such a proof, Wintenberger's stable likelihood, bounds the recursion at the
smallest variance it can reach. So it excludes persistent processes (`β`
near 1) whose filter does contract.

**The start.** The first `start_rows` rows of a target, or the rows of the
first `start_span` of the clock, are fitted by the model's own projected
Newton on their likelihood, from a small grid, and replayed at the fit to
build `H`. They report nothing, as `below_warm_rows`. The Newton runs until
a step is at the parameters' rounding, so where it stops is not a matter of
`ln`'s last bits. A fit is at its optimum when the gain its Newton step
predicts inside the process's bounds is at most `10⁻⁶` nats. Where the step
stays inside, that gain is the Newton decrement `½ gᵀ M⁻¹ g`, with `M` the
observed curvature; where it does not, the step is projected onto the
bounds in `M`'s metric. So an optimum on a bound, a GJR whose `α` is 0, is
one.

**It reports only a process its rows identify.** On returns with no
conditional variance the likelihood is flat along a ridge, where `β` is not
identified once `α` is 0, and any fit there is noise. So a start's fit is
stepped from only when its likelihood-ratio statistic against a constant
variance, `2 (l_fit − l_const)` with the weights at Kish's count, reaches
χ² at 0.999 for the restrictions a constant variance puts on the process.
That is 13.8 for `garch` (`α = β = 0`), and 16.3 for `gjr`, `egarch` and
`figarch` (three each); each input adds one, `π = 0`, so `garch` with one
input waits for 16.3. `figarch`'s `d = 0` and `φ = β` leave its four
parameters one variance, `ω/(1 − β)`, so three are the process's. `π = 0` is where an input that explains nothing
sits, so a fit with a `π` at 0 is no bound that makes a start wait. A
start with inputs also waits while its rows do not tell the inputs apart:
an input constant over them is `ω` again, and two that move together are
one. It waits where the start's information in the variance equation's
parameters, scaled to a unit diagonal, fails the relative pivot rule the
library's other definite solves read a system by. A pivot of its Cholesky
factorization at or under `10⁻¹⁰` of the largest diagonal entry fails it.
Until one does, its rows are withheld as `not_identified`, as are the
rows of a start that holds only zeros. A fit that passes but is not at its optimum is never
stepped from either; its wait is the start's own, `below_warm_rows`. A try
that does not start keeps its rows and fits again once as many new rows
have come as it held. That is twice its rows while `max_state_bytes` holds
them (each target's share of it), then the latest rows the share holds, the
oldest leaving as each new one comes. No count of tries ends the wait, and
nothing is cleared, which could not cure a shortage of evidence.

**Variance targeting** (`variance_targeting=True`, `vol="garch"` with
normal innovations) sets the level by the running mean square, `ω = v̄ (1 −
α − β)`, and learns `α` and `β` alone, with a Huber-clipped score. It is
another estimate, `ω` a moment rather than a likelihood one, and needs
returns with a finite fourth moment. It is not the safer choice. On series
at persistence 0.98 its forecasts, each made before its row, lost a mean of
3.4 nats over 8,000 rows to the true process, against 2.8 for the free `ω`:
its level rides on one running mean the whole stream shares.

The state is `8 (2k + q + m + n(n + 1)/2 + 9)` bytes a target for `k`
parameters in the variance equation, `q` of them the inputs' `π` (whose
running mean adds the `q`), `m` shape parameters and `n` stepped
coordinates. EGARCH keeps
64 bins of residuals beside them, 512 bytes. That is 168 bytes for `garch`,
216 for `gjr`, 728 for `egarch`, 208 for `garch` with a t, and `216 + 8 M`
for `figarch` (8,216 at 1,000 lags). The start's buffer adds 16 bytes a
row until it is fitted (8 more for each input), 32,000 at the default, and after a try that does
not start up to the target's share of `max_state_bytes`. For `figarch` the
buffer and the ring are never held together, so the target takes the
larger of the two, `216 + max(16 rows, 8 M)`: a buffer of up to 500 rows
at 1,000 lags adds nothing. `garch` with two inputs holds 288 bytes a
target.

#### Forecasts at horizons

`forecast_rows` asks for the variance further ahead. Each horizon is a count of
rows `h` from the row read, its own included. It gives the variance `h − 1`
rows past this one, `variance_h<h>_<t>`, and the sum of the variances over
those `h` rows, `variance_sum_h<h>_<t>`. That sum is the variance of the
`h`-row return, the quantity an option's implied variance over the same span
is set beside.
Horizon 1 is `variance_<t>` itself, the same number. Both are read before the
row is learned, from the rows before it, so a forecast never reads a row it
forecasts. A dict names the horizons, `forecast_rows={"week": 5, "month": 21}`
giving `variance_month_<t>`.

The recursion runs ahead from the row's variance, each later row's `e²`
read as its variance and GJR's asymmetric term as half of it, as `arch`'s
analytic forecast reads them:

```
garch, gjr  f_1 = s_t,  f_h = ω + α f_{h−1} + γ f_{h−1}/2 + β f_{h−1}
egarch      f_h = exp(L_h) Π_{j=1..h−1} M(β^{j−1}),  L_1 = ln s_t,  L_h = ω + β L_{h−1}
            M(c) = E exp(c (α (|z| − √(2/π)) + γ z))
figarch     f_h = ω/(1 − β) + Σ_{j=1..M} λ_j e²_{t+h−1−j},  a forecast for each row at or after t
sum         F_h = f_1 + … + f_h
```

With inputs, horizon 1 reads the row's own `x_t`, and every later row
holds them at `x̄`, their weighted mean over the rows learned, the start's
included: no input past the row is known. So `ω` above reads `ω̄ = ω +
π'x̄` past the row, and the forecasts revert to the long-run variance the
inputs' mean gives.

GARCH and GJR revert geometrically to `ω/(1 − P)`, `P = α + γ/2 + β`.
EGARCH's is Nelson's closed form for normal innovations. The log-variance is
linear in the news terms, which are independent from row to row. So the
variance's mean is the exponential of the log-variance's mean times the
product of the news terms' moment generating function,
`M(c) = e^{−cα√(2/π)} [e^{(cα+cγ)²/2} Φ(cα + cγ) + e^{(cα−cγ)²/2} Φ(cα − cγ)]`.
`arch` makes it only by simulation, and agrees within its sampling error.
The forecasts are not held within the bounds the filter holds the row's own
variance to.

One pass from horizon 1 to the largest makes every horizon and carries the
sum, so no horizon is summed twice; the sum is compensated, the variances'
own to a rounding. The pass costs the largest horizon times one step's
work: 4 multiply-adds for `garch` and `gjr`, `truncation + 2` for `figarch`,
and 164 for `egarch`, whose step calls `exp` and `erfc` five times, each
counted as 32. A spec whose forecasts would take more than
`max_horizon_work` (2^20 a target a row by default) is refused, naming the
work: that holds `garch` to horizons of 262,144 rows, `egarch` to 6,393 and
`figarch` at 1,000 lags to 1,046. The forecasts add nothing to the state.

#### Rows and the clock

**The recursion is per observation.** A row's lags are the row before it,
however far back on the clock that was, so a gap, a capped gap or a session
change breaks nothing. None of the process's parameters has a duration
form: `β` is a share of the last observation's variance, not of an hour's. The start is the one span: of rows, `start_rows`, or of the clock,
`start_span`, a number of the clock column's units or a duration such as
`"1d"` on a temporal clock. A span's rows are the data's to decide, so their
bytes are checked as they come: a warning past half of `max_state_bytes`,
and a refusal at it. The horizons count rows and have no clock form: the
recursion steps once a learned row, so a span of the clock holds as many
steps as the data puts in it. A row of weight 0 is no step, so it is in no
horizon, and it reads the forecasts the next learned row reads.

#### Example

**It recovers the process it did not see made.** This code simulates a
GARCH(1,1) at `(ω, α, β) = (0.05, 0.08, 0.9)` and prints the fit after
20,000 rows:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# Simulate the process: each row's variance from the row before.
rng = np.random.default_rng(7)
e, s, r = 0.0, 2.5, []
for z in rng.standard_normal(20_000):
    s = 0.05 + 0.08 * e * e + 0.9 * s
    e = np.sqrt(s) * z
    r.append(e)
pl.DataFrame({"r": r}).write_parquet("returns.parquet")

# Fit it, a row at a time after the start of 2,000 rows.
model = rp.spec.garch("g", targets=["r"])
bank = rp.ModelBank([model])
bank.fit(pl.scan_parquet("returns.parquet"))

# Read the fit's terms by name and compare them with the truth.
coef = {row["term"]: row["coef"] for row in bank.coef().iter_rows(named=True)}
print(f"alpha: true 0.08, found {coef['alpha']:.2f}")
print(f"beta: true 0.90, found {coef['beta']:.2f}")
```

```text
alpha: true 0.08, found 0.07
beta: true 0.90, found 0.91
```

#### See also

For a variance or a clustering measured another way:

- [`ew_cov`](#ew_cov--exponentially-weighted-moments), for a variance with a
  fixed decay rather than a fitted one.
- [`hawkes`](#hawkes--how-much-does-each-event-excite-the-next), for the
  clustering of events rather than of sizes.

#### Notes

**It agrees with `arch`'s batch fit to first order.** It is the recursive
prediction-error method: with the gain `1/W` and a batch start, its
estimate agrees with the batch fit of the same rows to first order. Against
`arch`'s batch fit of 10,000 simulated rows, on 50 series of each form,
every parameter was within two of `arch`'s standard errors on all 50. The
log-likelihood its forecasts lost to `arch`'s path had a median of 0.9 of
its expected size, `(k/2)(ln(n/n₀) + 1)` for `k` parameters and `n₀` rows of
start, and was at most 2.6 times it. At 100,000 rows the median gap was 0.04
to 0.14 standard errors.

**Left out.** GARCH(2,2): `arch`'s own fits split its two `β`s differently
from series to series, and the online fit is not yet held to a tolerance.
GJR with the skewed t: under skewed innovations GJR's persistence reads
`E[z² 1(z < 0)]` rather than the half of `E[z²]` its forecast uses. A t on
EGARCH or FIGARCH, APARCH, the generalized error distribution and a mean
model: none has an online estimator checked against `arch`. A t on EGARCH
would also have no variance forecast past one row: `E e^{c|z|}` is infinite
for a t.

### `dcc` — how do the correlations move from row to row?

*API:* [`rheon.spec.dcc`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.dcc) — *Rust:* [`dcc.rs`](../crates/rheon-core/src/dcc.rs) — *Outputs:* [fields](OUTPUTS.md#dcc)

Dynamic conditional correlation (Engle 2002) and its asymmetric form
(Cappiello, Engle and Sheppard 2006): the full correlation matrix of `k`
return series given the rows before the row, with two or three numbers that
drive it, given or learned one row at a time.

Use `dcc` on a few return series whose correlations change with time, and
rise after joint falls, when you want each pair's correlation forecast
before the row arrives: a portfolio's risk, a hedge ratio, a pair's spread.
Each series is standardized by its own GARCH(1,1) margin, as `garch` would
fit it. The margins' start (2,000 rows by default) reports no variance, and
the correlation's own start (500 rows) after it reports no correlation;
after them every row reports each series' variance, each pair's
correlation and the row's log-density under it.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.dcc),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#dcc). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes),
apart from `features`, which it refuses. `half_life` or `lam` is required,
`"inf"` for no decay. A row's weight scales its evidence. A row of weight 0,
or with any series null, steps nothing, and is still read.

#### The recursion

With `z_t` the row's standardized residuals, each series' return over the
square root of its margin's variance read before the row, and `n_t` their
negative parts:

```
Q_{t+1} = (1 − a − b) Q̄ − g N̄ + a z_t z_tᵀ + g n_t n_tᵀ + b Q_t
R_t     = diag(Q_t)^{−1/2} Q_t diag(Q_t)^{−1/2}
Q̄ ← Q̄ + (w/W)(z zᵀ − Q̄),   N̄ ← N̄ + (w/W)(n nᵀ − N̄),   W ← λ W + w
```

`R_t` is the correlation reported for row `t`. The targets `Q̄` and `N̄` are
the weighted means of the rows learned, so the level the correlations
revert to is the stream's own. `g = 0` is DCC(1,1): set `asymmetric=True`
for the asymmetric model, whose correlations also answer joint negative
returns. With `margins="none"` the targets are taken as standardized
residuals already.

**The set it holds** is `a, b, g ≥ 0` and `a + b + δ g ≤ 1 − 10⁻⁴`, with `δ`
the largest eigenvalue of `Q̄^{−1/2} N̄ Q̄^{−1/2}`. That is exactly where the
intercept `(1 − a − b) Q̄ − g N̄` is positive semi-definite, so every `Q_t` is
positive definite and every `R_t` a correlation matrix. It is also where a
covariance-stationary `Q_t` exists, given innovations with a finite fourth
moment: the recursions of `Q`'s mean and variance have the matrices `b I`
and `b² I`, so `δ` enters only through positivity. A Student-t margin
fitted at or below 4 degrees of freedom has no fourth moment, and the bank
says so once. The targets move, and `δ` with them, so the set is held at
every row: parameters the targets have left outside are projected back onto
it before the next `Q` is formed. Given parameters are projected afresh
each row, so they return as given once the targets do.

#### The update

Each learned row of weight `w` steps the parameters by the score of its
quasi-log-likelihood `l_t = −½ (ln |R_t| + z_tᵀ R_t⁻¹ z_t)`. The score
reads `∂R` from `∂Q`, and its own recursion carries `∂Q` at the parameters
of its time:

```
r = w / W
H ← H + r (I_t − H)          I_t = ½ tr(R⁻¹ ∂R_i R⁻¹ ∂R_j)
θ ← Proj_H(θ + r H⁻¹ g_t)    g_t = ½ (uᵀ ∂R_i u − tr(R⁻¹ ∂R_i)),  u = R⁻¹ z
```

`Proj_H` is the nearest point of the set in `H`'s own metric, as `garch`
projects. With no decay `r = 1/(n + N)` after a start of `N` rows; under a
half-life it floors at `1 − λ`, a tracking estimator whose targets and
information forget at the same rate.

**The start.** The correlation's first `start_rows` rows after the
margins', or the rows of its first `start_span` of the clock, are buffered.
Their weighted means are the targets, and the parameters are the maximum of
their batch quasi-likelihood with `Q_1 = Q̄`, by the model's own projected
Fisher scoring from a small grid, run to its rounding. The buffer is then
replayed at the fit to build `H`.

**A start reports only a recursion its rows identify.** The test is the
condition itself, not a count of rows. The fit is withheld unless its
likelihood-ratio statistic against constant correlation (`a = b = 0`, and
`g = 0`), with the weights at Kish's count, reaches χ² at 0.999 for those
restrictions: 13.8 for DCC's two and 16.3 for the asymmetric model's three.
The information there must be positive definite too, and a buffer whose
`Q̄` is not positive definite is withheld the same way: fewer rows than
series, a series constant over them, or two the same. `Q̄` is judged by its
conditioning, a condition number past `1 / (k ε)` refused, because a `Q̄`
short of rank has an exact 0 eigenvalue that the factor computes with
rounding of either sign. A fit that passes is stepped from only at its
optimum: the gain its projected Newton step predicts is at most `10⁻⁶`
nats, which an optimum on a bound of the set meets as well. A start that
fails either keeps its rows and fits again once as many new rows have come
as it held: twice its rows while `max_state_bytes` holds them, then the
latest rows the cap holds, the oldest leaving as each new one comes. No
count of tries ends the wait, and nothing is cleared, which could not cure
a shortage of rows. While a fit is withheld, `withheld_reason` reads
`not_identified`; while it is short of its optimum alone, the start's own
warm-up, `below_warm_rows`. With `a` and `b` given there is nothing to
identify, and the start waits only for a `Q̄` that is positive definite.
Under a decay the evidence a buffer holds stops growing: its rows' weights
are decayed to the last, so Kish's count of them is at most about `(1 +
λ)/(1 − λ)`, 289 at a half-life of 100 rows. Give a half-life several times
the rows that tell the recursion from a constant correlation.

The state is `32 k² + 96` bytes for DCC and `48 k² + 144` for the
asymmetric model, learned (`16 k² + 64` and `24 k² + 72` given), for `k`
series, and the margins' 168 bytes a series. Each start's buffer adds its
own until it is fitted: 16 bytes a row a series for the margins', `8 (k +
2)` a row for the correlation's, which a start that waits grows up to what
`max_state_bytes` holds. The margins' start comes first, so with
the default starts `max_state_bytes`' 256 KB holds eight series. A row
costs `O(p k³)` for `p` parameters: the factor and inverse of `R`, and `R⁻¹
∂R` for each parameter.

#### Rows and the clock

**The recursion is per observation**, as a GARCH's is: `Q_{t+1}` follows
row `t` however far back on the clock it was. The decay acts on the clock,
through the targets, the weight and the estimator's gain, so a half-life in
clock units forgets the same at any row rate. Each start is a span of rows,
`start_rows` and `margin_start_rows`, or of the clock, `start_span` and
`margin_start_span`, a number of the clock column's units or a duration
such as `"1d"` on a temporal clock. A span's rows are the data's to decide,
so their bytes are checked as they come: a warning past half of
`max_state_bytes`, and a refusal at it.

**No rolling window.** The targets could be read over a window, but the
estimate could not: its sensitivities and information are the recursion's
whole path, from which no row can be taken back out. A half-life gives the
targets and the estimate the same memory.

#### Example

**It recovers the process it did not see made.** This code simulates two
series whose correlation follows a DCC at `(a, b) = (0.04, 0.93)` and prints
the fit after 20,000 rows:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# Set the truth: the parameters and the level the correlations revert to.
rng = np.random.default_rng(3)
a, b, qbar = 0.04, 0.93, np.array([[1.0, 0.5], [0.5, 1.0]])

# Simulate the process: each row's correlation from the rows before.
q, rows = qbar.copy(), []
for _ in range(20_000):
    d = np.sqrt(np.diag(q))
    z = np.linalg.cholesky(q / np.outer(d, d)) @ rng.standard_normal(2)
    q = (1 - a - b) * qbar + a * np.outer(z, z) + b * q
    rows.append(z)

# Write the two series to a file.
z = np.array(rows)
pl.DataFrame({"x": z[:, 0], "y": z[:, 1]}).write_parquet("z.parquet")

# Fit it, a row at a time after the start of 500 rows.
model = rp.spec.dcc("c", targets=["x", "y"], margins="none", half_life="inf")
bank = rp.ModelBank([model])
bank.fit(pl.scan_parquet("z.parquet"))

# Read the fit's terms by name and compare them with the truth.
coef = {row["term"]: row["coef"] for row in bank.coef().iter_rows(named=True)}
print(f"a: true 0.04, found {coef['a']:.2f}")
print(f"b: true 0.93, found {coef['b']:.2f}")
```

```text
a: true 0.04, found 0.04
b: true 0.93, found 0.93
```

#### See also

For correlations measured another way:

- [`garch`](#garch--how-large-will-the-next-return-be), for each series'
  variance alone.
- [`deco`](#deco--one-correlation-for-the-whole-matrix), for one average
  correlation across many series, at `O(k)` a row.
- [`ew_cov`](#ew_cov--exponentially-weighted-moments), for a correlation
  with a fixed decay rather than a fitted recursion.

#### Notes

**It agrees with the batch fit to first order.** It is the recursive
prediction-error method: with the gain `1/W` and a batch start, its
estimate agrees with the batch fit of the same rows to first order. No
Python library fits DCC, so the batch fit it was held to is Engle's
definition written out, its targets the sample means, maximized by scipy's
SLSQP, with the second stage's sandwich standard errors. It was run on 50
simulated series of 10,000 rows for each of six forms: two and three
series, DCC and the asymmetric model, normal and Student-t(5) innovations.
On every form every parameter was within two of those standard errors on
all 50 series. The largest gap was 1.17 of them under normal innovations
and 1.95 under t(5), and the median gap between the two correlation paths
over the second half of the rows was at most 0.017. Through GARCH margins, against
`arch`'s margins and the same second stage, ten series of two forms were
within 1.2 standard errors.

**Two stages.** As in the batch method, the margins' estimation error is
not carried into the correlation's: the quasi-likelihood is the second
stage's alone, and the margins are `garch`'s, unchanged.

### `realized` — how much did it move, and did it jump?

*API:* [`rheon.spec.realized`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.realized) — *Rust:* [`realized.rs`](../crates/rheon-core/src/realized.rs) — *Outputs:* [fields](OUTPUTS.md#realized)

Realized volatility, measured three ways, and two tests for a jump, for
each series of returns, one pass.

Use `realized` on log returns of a price sampled at intervals, when you
want to know how much a series has moved and whether a move was a jump.
The realized variance counts every move. Bipower and tripower variation
leave the jumps out, and the difference between them is the jump part.
Barndorff-Nielsen and Shephard's ratio test says whether the stretch the
measures cover holds a jump; Lee and Mykland's statistic says whether this
row's own return is one, against the volatility of the returns before it.
Each target is its own series. Set `session` with `session_gap="reset"` to
measure a day at a time, or a window, or a half-life, to measure the
recent past.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.realized),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#realized). The
parameters every spec takes are in [What every spec takes](#what-every-spec-takes),
apart from `features`, which it refuses. A row's weight counts its return
that many times. A row of weight 0 counts in no clock, and its return is in
no product. A null return is that target's alone, and its products reach
across the row to its last return. A gap past `gap_cap` or a session's
edge is a step of the clock like any other, its products reaching across
it; `session_gap="reset"` measures each session on its own.

#### The measures

For `n` returns `r_j`, `a_j = |r_j|` and `μ_p = E|Z|^p` of a standard
normal (Barndorff-Nielsen and Shephard 2004, 2006; Lee and Mykland 2008):

```text
RV  = Σ r_j²
BV  = μ_1^-2 Σ_{j≥2} a_j a_{j-1}                      μ_1^-2 = π/2
TV  = μ_{2/3}^-3 Σ_{j≥3} (a_j a_{j-1} a_{j-2})^(2/3)
TQ  = n μ_{4/3}^-3 Σ_{j≥3} (a_j a_{j-1} a_{j-2})^(4/3)
bns = √n (1 - BV/RV) / √(ϑ max(1, TQ/BV²))      ϑ = π²/4 + π - 5
lm  = r_i / σ_i,   σ_i² the mean of a_j a_{j-1} over the products before row i
```

`RV` converges to the integrated variance plus the squared jumps, `BV` and
`TV` to the integrated variance alone. `bns` is a standard normal without
jumps and large and positive with them. `√(2/π) · lm` is a standard normal
without a jump, so it reads as a z-score. Every field but `lm` and
`lm_jump` is of the returns before the row, so a row's own return is in its
statistic and in nothing it is tested against.

**No small-sample factor.** The papers have none, so neither has
`realized`: no `n/(n-1)` on `BV`, no `n/(n-2)` on `TQ`. Libraries differ
here, and the [tested against](#tested-against) note below says by how
much.

**A window of `W` returns holds `W - 1` products and `W - 2` triples**, as
the papers' sums over `W` returns do. A product is in the window only while
both of its returns are, so a window's measures are the papers' on its
returns.

**The count `n`.** `bns` and the flag read the count of returns the
measures are of: Kish's effective count `(Σ w)² / Σ w²` of the returns'
weights, decayed or in a window, which is the count itself at equal weights
without a decay. `TQ`'s factor is the weight `Σ w` instead, the length of
the period its sum is of, so `TQ/BV²` is 1 at a constant volatility whatever
the weights. With unequal weights `1 - BV/RV` has the variance `ϑ Σ w² / (Σ
w)² = ϑ / n`, so with these two factors `bns` is a standard normal without
jumps under a decay as well. A simulated null of 20,000 streams under a
half-life of 100 rows gives a standard deviation within 0.035 of 1, and a
size at the 1% cut between 0.008 and 0.016. The sums themselves are
weighted, so they scale with the weights; `n`, `bns` and the flag do not.

**Lee and Mykland's flag.** It is the paper's test of the largest `|lm|`
among `n` returns, at confidence `lm_beta` (0.99, a test at 1%):

```text
C_n = √(2 ln n)/c - (ln π + ln ln n) / (2 c √(2 ln n))     S_n = 1 / (c √(2 ln n))
lm_jump = (|lm| - C_n) / S_n > -ln(-ln lm_beta)            c = √(2/π)
```

`lm_window = K` takes `σ_i` from the target's `K - 1` returns before the row
alone, the paper's window; without it, `σ_i` is read from the measures'
own window or decay.

#### The forms

The measures take one of four forms:

- **One pass** (`half_life = inf`): the measures of every return so far.
- **A decay** (`half_life`): each term decays by `0.5 ** (Δclock /
  half_life)` from its row, the default form.
- **A window of rows** (`window_rows`): the last `W` rows of positive
  weight. Its bytes, `16 + 56 m` a row for `m` targets, are known from the
  settings, so a window past `max_state_bytes` is refused when the spec is
  built, naming the largest that fits.
- **A window of the clock** (`window_size`): the rows of the last span,
  `closed` deciding the row exactly that old. The data decides how many
  rows it holds, so its bytes are checked as the rows go in: past half of
  `max_state_bytes` a warning names the group once, and the row that would
  pass the cap stops the run.

A window forgets by dropping, so it takes no decay: set `half_life` to
`inf` beside one. Without a window a group holds `8 + 144 m` bytes,
whatever the stream's length.

#### Tested against

The measures and both tests are held to R's `highfrequency` (`rRVar`,
`rBPCov`, `rMPVar`, `rTPQuar`, `BNSjumpTest`, `intradayJumpTest`) and
`yuima` (`bns.test`), the field's reference implementations. RV and BV are
held to Python's `hfvol` too. The days are simulated, with and without
planted jumps, and Binance's BTCUSDT minutes and ETCUSDT trades.
`highfrequency` puts `N/(N-2)` on its tripower measures, and `yuima` takes
the number of prices `n + 1` where the paper has the `n` returns. With
those factors divided out every measure agrees to a few units in the last
place, and each test to the rounding of the difference it is built on.
`intradayJumpTest` agrees on `√(2/π) · lm` with `dontIncludeLast = TRUE`,
which leaves the row's own return out of its volatility as the paper does;
its default puts it in.

#### See also

Where `realized` is not the measure you need:

- [`rcov`](#rcov--a-blocks-realized-covariance-robust-to-noise), for a
  block's realized covariance robust to microstructure noise.
- [`garch`](#garch--how-large-will-the-next-return-be), for a conditional
  variance fitted to the returns.

### `power_variation` — power variation and kernel-weighted autocovariance

*API:* [`rheon.spec.power_variation`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.power_variation) — *Rust:* [`power_variation.rs`](../crates/rheon-core/src/power_variation.rs) — *Outputs:* [fields](OUTPUTS.md#power_variation)

Sums of powers of returns and of their kernel-weighted autocovariances,
for each series of returns, one pass: the building block the realized
measures are made of.

Use `power_variation` on returns when the measure you need is one of these
sums, or is composed from them in Polars. Each term is named, and its name
names its fields:

- **A multipower term** (`powers`): the sum of products of powers of
  consecutive absolute returns, oldest first. `[2]` is the realized
  variance, `[1, 1]` the bipower sum, `[4/3, 4/3, 4/3]` the tripower
  quarticity's; with `normalize` each is Barndorff-Nielsen and Shephard's
  multipower variation, `W^(Σp/2 - 1) / Π μ_p` times the sum. `W` is the
  returns' weight, which is their count `n` at unit weights without a
  decay.
- **A kernel term** (`kernel`, `bandwidth`): `Σ_j [r_j² + 2 Σ_h c_h r_j
  r_{j-h}]` for lags up to the bandwidth `H`, with the Bartlett, Parzen,
  Tukey-Hanning or rectangular kernel at `c_h = k(h/(H + 1))`, Newey and
  West's weights for the Bartlett kernel. `flat_top` gives lag 1 the full
  weight, `c_h = k((h - 1)/H)`, the flat-top realized kernel, which takes a
  bid-ask bounce out of a realized variance. `center` takes each product
  about the mean, which is the HAC long-run variance times `n`.
- **A scale** (`scale = K`) on either: the term of the `K`-step returns
  at every row, divided by `K`, which averages it over the `K` offset grids.
  The sum of squares at two scales is what two-scale realized variance is
  composed from: `sub_K - (n̄_K / n̄_J) sub_J`, `n̄_K = (n - K + 1) / K`.

Each target reports `n` beside its terms, the count of returns they are of:
Kish's effective count `(Σ w)² / Σ w²` of their weights, decayed or in the
window, which is the count at equal weights. No term carries a
small-sample factor such as `n/(n - h)`; divide or multiply in Polars
where a convention needs one. Its forms are
`realized`'s: one pass, a decay, a window of rows (checked against
`max_state_bytes` when the spec is built) and a window of the clock
(checked as the rows go in); a term is in a window while every return in
it is. Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.power_variation).

### `tdigest` — where do this column's quantiles stand?

*API:* [`rheon.spec.tdigest`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.tdigest) — *Rust:* [`tdigest_model.rs`](../crates/rheon-core/src/tdigest_model.rs) — *Outputs:* [fields](OUTPUTS.md#tdigest)

Quantiles of each target column, read at the levels you list from a merging
t-digest whose error is uniform in rank, each beside the mean of the tail past
it: the value at risk and the expected shortfall. Each row reports every
level from the rows before it, so a row's own value is never in its own
quantile.

Use it for a column whose values sit near 0 or take both signs (returns,
residuals, spreads), where a sketch of relative accuracy misreads. Each
target has its own digest in each group. The digest never forgets: for a
quantile that does, use `ewm_quantile` with a `window_size`, the windowed
DDSketch, whose buckets drop a row as it leaves the window.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.tdigest),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#tdigest). The
parameters every spec takes are in [What every spec
takes](#what-every-spec-takes), apart from `features`, `half_life` and
`lam`, which it refuses. It takes no window either: a merging digest folds
each row into a centroid's mean and cannot take it out again.
A row's weight is its mass in the digest, and scaling every weight changes
nothing but `weight_sum`. A row of weight 0, or whose value is null, adds
nothing; it is still read.

#### The update

Each target `t` keeps a merging t-digest `D_t` (Dunning and Ertl, 2019, on
the scale function `k(q) = δ/(2π) asin(2q − 1)`), with the weight `W_t` and
the squared weights `S_t` of the rows it holds. For a row of weight `w` and
value `y`:

```text
q_p,t  = D_t.quantile(p)        read first: the rows before this one
es_p,t = mean of Q_t over the tail past q_p,t
D_t    ← D_t + (y, w)           when w > 0 and y is present
W_t    ← W_t + w,   S_t ← S_t + w²
```

The digest keeps centroids ascending by mean, each spanning at most 1 in
`k`, and merges a buffer of `8δ` rows into them. A read merges a copy, so
the state is the rows' alone, and interpolates between the centroids'
centers from the smallest row to the largest. The merge's limit `k⁻¹(k(q) +
1)` is taken through the sine addition formula with `sin` and `cos` of `2π/δ`
summed from their series, so no math library's last bit decides a merge and
the digest is the same bits on every platform.

**The tail's mean comes from the digest's own centroids.** A level below
0.5 reads the lower tail, `E[X | X ≤ q]`, and one above it the upper,
`E[X | X ≥ q]`; 0.5 itself has no tail of its own and is refused by name.
The tail holds the mass `m = p W_t` from the smallest mean, or
`m = (1 − p) W_t` from the largest. Each centroid wholly inside counts at its
mean, and the one the level falls in is split, counted at only the weight
the tail still needs:

```text
es_p,t = (Σ_{i<j} m_i c_i + (m − M_{j−1}) c_j) / m
```

The values keep their sign (a loss is not negated), and it is null where
its level is. On 100,000 rows of the four streams below it was within 4.3%
of the exact tail mean at `δ = 100` and 1.2% at 200. A mean of the read's
interpolated curve was tried first and is not used: its end pieces run to
the smallest and largest row over half a centroid's weight, and on a heavy
tail that took the mean as far as 97% off.

**A level is withheld until its rows identify it.** Take Kish's effective
size `n_t = W_t²/S_t`, the rows themselves for rows of equal weight. The
`i`-th smallest of `n` rows sits at level `i/(n + 1)` in expectation, so a
level past `1/(n + 1)` or `n/(n + 1)` lies beyond the smallest or largest
row, where the digest can read only the extreme. Such a level is null and
`withheld_reason` is `level_beyond_sample`, while

```text
p (n_t + 1) < 1   or   (1 − p) (n_t + 1) < 1
```

Levels of 0.25 and 0.75 are reported from the fourth row, 0.05 and 0.95
from the twentieth, 0.01 and 0.99 from the hundredth; no count of rows is
fixed.

**State.** At most `δ + 1` centroids and `8δ` buffered rows a target, 16
bytes each, plus a fixed header: about 14.5 KB a target at `δ = 100` and
29 KB at 200, whatever the stream's length. A spec past `max_state_bytes`
(256 KB by default) is refused at the call.

#### Notes

**Its rank error, against exact ranks and the `tdigest` package.** It was
measured at 100,000 rows of normal, lognormal, Student-t(3) and Pareto(2.5)
values over 20 seeds, at the levels 0.01, 0.05, 0.5, 0.95 and 0.99. The
largest rank error `|F(q̂) − p|` was 2.0e-3 at `δ = 100` and 9.3e-4 at 200,
held under 3.8e-3 and 1.8e-3. The `tdigest` package's own digest read the
same quantiles within those tolerances in rank. The error is largest at the
median and smallest in the tails, where `k_1` keeps the centroids small.

**A read costs a merge.** Each row's read merges a copy of the buffer into
the centroids, `O(δ log δ)` a target, so a digest of compression 100 costs
more a row than a running moment does.

### `ddsketch` — how large are this column's quantiles and tails?

*API:* [`rheon.spec.ddsketch`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ddsketch) — *Rust:* [`ddsketch_model.rs`](../crates/rheon-core/src/ddsketch_model.rs) — *Outputs:* [fields](OUTPUTS.md#ddsketch)

Quantiles of each target column, read at the levels you list from a DDSketch
whose error is relative to the value, each beside the mean of the tail past
it: the value at risk and the expected shortfall. Each row reports every
level from the rows before it, so a row's own value is never in its own
quantile.

Use it where a reading within 1% of the true quantile is what is asked,
whatever the value's size: latencies, sizes, prices, losses. Each target has
its own sketch in each group. It forgets by a window, exactly: `window_rows`
(the last rows of positive weight) or `window_size` (the rows of the last
span of the clock). Or it forgets by a decay, `half_life` or `lam`, rounded;
or it never forgets. For values near 0 or of both signs whose rank matters,
use `tdigest`.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ddsketch),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#ddsketch). The
parameters every spec takes are in [What every spec
takes](#what-every-spec-takes), apart from `features`, which it refuses,
and `half_life` and `lam`, which it refuses beside a window. A row's weight is its mass in the sketch, and scaling every weight
changes nothing but `weight_sum`. A row of weight 0 counts in no clock and
is in no window.

#### The update

With `γ = (1 + α)/(1 − α)`, a value `x > 0` goes to the bucket
`k = ⌈log_γ x⌉`, which covers `(γ^(k−1), γ^k]` and reads `2γ^k/(1 + γ)`,
within `α` of every value in it. This is Masson, Rim and Lee's sketch (2019)
with the `ddsketch` package's logarithmic mapping. A negative value goes to a second
store by its magnitude, and one within about `2.2e-308` of 0 to a count of
its own. For a row of weight `w` and value `y`:

```text
q_p,t  = the representative of the first bucket, in value order, whose weight reaches p W_t
es_p,t = (Σ_{k<j} c_k r_k + (m − C_{j−1}) r_j) / m
D_t    ← D_t + (y, w)           when w > 0 and y is present
W_t    ← W_t + w,   S_t ← S_t + w²
```

The tail is the lower one below a level of 0.5, `m = p W_t` from the
smallest value, and the upper one above it, `m = (1 − p) W_t` from the
largest; 0.5 itself is refused by name. The bucket the tail ends in counts
only the weight the tail still needs, so the tail's mean is within `α` times
the tail's mean magnitude of the exact one, and the quantile within `α` of
the exact quantile (numpy's `inverted_cdf` over the rows). The logarithm and
the power are summed from their series in plain arithmetic, so a row's
bucket is the same on every platform.

**A window takes a row out exactly.** A ring keeps each row's weight and, per
target, its bucket; a row leaving takes its weight out of its bucket, and a
bucket whose rows have all left is removed, so the sketch is the one the
window's rows alone build. A clock window's rows leave at the first row at
or after the one they age out on, before that row's read; `closed` decides
whether a row exactly `window_size` old stays. Under a window, `weight_sum`
is the weight of the rows the row reads.

**A decay is rounded.** Under `half_life` or `lam` each row, after its read,
multiplies every bucket's weight, the zero count and the weights by its
clock step's factor `f` before it adds itself, each multiply and add rounded
once. A bucket formed `k` rows ago then holds its exact decayed sum within a
relative `2ku/(1 − 2ku)`, `u = 2⁻⁵³`: about `2.2e-12` after 10,000 rows. The
tests hold each bucket to its exact sum in fractions, from the same factor,
within that bound. No row can be taken out of such a weight exactly, which is
why the window is the form that drops one. A bucket that decays to 0 is
removed. The factor is the platform's, so a decayed sketch agrees across
platforms within its last bits.

**A level is withheld until its rows identify it**, by `tdigest`'s rule:
null while `p (n_t + 1) < 1` or `(1 − p)(n_t + 1) < 1`, `n_t` Kish's
effective size of the rows, with `withheld_reason` `level_beyond_sample`.

**State.** Each store keeps its live buckets alone, 12 bytes each (16 under
a window, with a row count): at most the data's log-range over `ln γ`,
about 1,470 a store for values spanning `1e-5` to `1e5` at the default `α`,
whatever the stream's length. A row window adds `window_rows (8 + 4m)` bytes
for `m` targets, checked at the call; the buckets and a clock window's rows
are checked against `max_state_bytes` as the rows go in, said past half of
it and refused at it.

#### Notes

**Against the `ddsketch` package and exact reads.** The streams were normal,
lognormal, Student-t(3) and Pareto(2.5), 100,001 rows each over 20 seeds.
Every bucket's key and count was the package's, and each representative and
each quantile was the package's within 2 ulps. Each quantile was within `α`
of the exact quantile, and each tail mean within `α` times the tail's mean
magnitude of the exact tail mean, with weights of 1 and of many sizes.

## Clustering and classification

These models put a label on each row: a cluster found in the features alone,
or a class learned from a label column. Every label, distance and probability
is read before the row is learned from, so it is out-of-sample like every
prediction here.

### `kmeans` — exponentially weighted k-means

*API:* [`rheon.spec.kmeans`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.kmeans) — *Rust:* [`cluster/kmeans.rs`](../crates/rheon-core/src/cluster/kmeans.rs) — *Outputs:* [fields](OUTPUTS.md#kmeans)

Exponentially weighted k-means over the feature columns.

Use `kmeans` when you know the number of clusters and they are round: it
labels each row with the nearest of `n_clusters` centers, before the row is
learned. Use [`micro`](#micro--density-based-clustering-any-shape) for
clusters of any shape or of a number you do not know, and
[`hmm`](#hmm--which-regime-are-we-in) where a row's cluster depends on the
one before it.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.kmeans),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#kmeans). The
parameters every spec takes are in [What every spec
takes](#what-every-spec-takes).

#### The update

Each center is the exponentially weighted mean of the rows assigned to it
(`ew_cov`'s mean recursion, per cluster). With `x` the row's features, `c_j`
center `j` and `n_j` its decayed weight:

```
j*    = argmin_j ‖x − c_j‖²          distances in units of each feature's EW sd (standardize=False: raw units)
n'_j  = λn_j                          every other center
n'_j* = λn_j* + w                     c'_j* = c_j* + (w/n'_j*)(x − c_j*)
```

Three rules keep the centers on the data: seeding, a split-merge move that
finds a cluster born after seeding, and a dead rule that re-places a center
whose cluster vanished. A row far outside its cluster moves neither `n_j`
nor `c_j`, and waits for the split-merge move. The docstring's Notes give
each rule, under *Seeding*, *Far rows, split and merge* and *Dead centers*.

#### Example

**The split-merge move gives a cluster born after seeding a center of its
own.** This code makes three round clusters, and a fourth that appears at
row 4,000. It fits `n_clusters=4` and counts each label's rows before the fourth
appears and at the end:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# Three round clusters, and a fourth that appears halfway through.
rng = np.random.default_rng(6)
n = 8000
centers = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 6.0]])
blob = rng.integers(0, 3, n)                                  # three clusters ...
late = np.arange(n) >= 4000
blob[late] = rng.integers(0, 4, late.sum())                   # ... and a fourth born at row 4000
xy = centers[blob] + rng.standard_normal((n, 2))
pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}).write_parquet("blobs.parquet")

# Label every row, then count each label's rows before the birth and at the end.
km = rp.spec.kmeans("km", features=["x", "y"], half_life=1000.0, n_clusters=4, warm_rows=500)
label = pl.scan_parquet("blobs.parquet").rheon.fit_predict([km]).collect()["km"].struct.field("cluster")
for lo, hi in ((3000, 4000), (7000, 8000)):
    counts = label[lo:hi].value_counts().sort("cluster")
    print(f"rows {lo}-{hi - 1}: " + ", ".join(f"cluster {c}: {k}" for c, k in counts.iter_rows()))
```

```text
rows 3000-3999: cluster 0: 322, cluster 1: 164, cluster 2: 338, cluster 3: 176
rows 7000-7999: cluster 0: 237, cluster 1: 266, cluster 2: 255, cluster 3: 242
```

Before the fourth cluster appears, clusters 1 and 3 share one of the three,
about 170 rows each. Once it appears, the split-merge move frees cluster 3
and places it on the new cluster, and each of the four holds about a quarter
of the rows.

#### See also

The guides to its settings:

- To choose `init` and `dead_frac`, see [Keep `kmeans` centers on the
  data](how-to/keep-kmeans-centers-on-the-data.md).

When the clusters are not round, not fixed in number, or labeled:

- [`micro`](#micro--density-based-clustering-any-shape), density
  clustering that finds the number of clusters and flags the rows in none.
- [`hmm`](#hmm--which-regime-are-we-in), Gaussian states whose next state
  depends on the last, seeded by `kmeans`' `init`.
- [`ew_class`](#ew_class--gaussian-classification-on-ew_cov-moments),
  Gaussian classes, one for each value of the column `label=` names.

#### Notes

**Seeding waits for `warm_rows` rows, places the centers, and replays the
rows.** [Seeding](explanation/warm-up-and-readiness.md#seeding-a-model-that-places-its-starting-values)
explains why the default takes the best of ten starts, measures it on
Fränti's benchmark sets, and gives what it costs each group.

**The default seeding places the centers scikit-learn's `KMeans` places.**
`init="lloyd"` is held to scikit-learn's `KMeans(n_init=10)` centers on the
same weighted rows, and `init="k-means++"` to scikit-learn's
`kmeans_plusplus` in distribution. On hundreds of thousands of rows, the
labels are held to the clusters that generated them and to scikit-learn's
`KMeans` on the same rows.

### `micro` — density-based clustering, any shape

*API:* [`rheon.spec.micro`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.micro) — *Rust:* [`cluster/micro.rs`](../crates/rheon-core/src/cluster/micro.rs) — *Outputs:* [fields](OUTPUTS.md#micro)

Density-based clustering over the feature columns, by small decayed
summaries linked into clusters.

Use `micro` when the clusters may have any shape, such as moons or rings,
or their number is not known. It flags the rows that belong to no cluster,
and follows clusters that appear and vanish. Use
[`kmeans`](#kmeans--exponentially-weighted-k-means) for a known number of
round clusters.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.micro),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#micro). [Choose
`micro`'s `eps`](how-to/choose-micro-eps.md) says how to choose `eps`, and
how a wrong value shows in the output. The parameters every spec takes are
in [What every spec takes](#what-every-spec-takes).

#### The update

It builds on DenStream's micro-clusters, with a linking step over them.
Each micro-cluster is a *summary*: a decayed weight `n`, a center `c`, and
a radius `ν`, the exponentially weighted root-mean-square distance of its
rows from the center. A summary is *established* once its weight reaches
`beta_mu` rows of the stream's mean row weight `w̄`. Distances are in units
of each feature's exponentially weighted standard deviation, floored by
`scale_floor` as [`kmeans`](#kmeans--exponentially-weighted-k-means)' are.
With `p` features, a row joins a summary only if the summary's radius stays
within `eps·√p`:

```
r²(j, v) = a ν²_j + a b ‖x − c_j‖²      a = n_j/(n_j + v),  b = v/(n_j + v)
n_j  ← λ n_j                                             every summary
j*   = the nearest established summary, if r²(j, w̄) ≤ eps² p,
       else the nearest summary not yet established, if r²(j, w̄) ≤ eps² p,
       else a new summary at x
ν²_j* ← min(r²(j*, w), eps² p)     c_j* ← c_j* + (w/(n_j* + w))(x − c_j*)     n_j* ← n_j* + w
```

Every `prune_every` units of the clock, or every `max_rows_between_prunes`
learned rows, the model drops the light summaries and links the established
ones: centers within `L` of each other share a label. The docstring's Notes
give the rules, under *Pruning*, *The prune schedule* and *The link*.

#### Example

**It finds two rings that no center-based rule can split, and flags
scattered rows as noise.** This code makes an inner and an outer ring, with
one row in twenty scattered over the square around them, and reads the last
4,000 rows:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# Two rings, an inner and an outer, as points in the plane.
rng = np.random.default_rng(7)
n = 10000
ring = rng.integers(0, 2, n)                                  # 0: the inner ring, radius 1; 1: the outer, radius 3
angle = rng.uniform(0.0, 2 * np.pi, n)
radius = np.where(ring == 0, 1.0, 3.0) + rng.normal(0.0, 0.1, n)
xy = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
# Scatter some rows as noise, and keep the truth to score against.
noise = rng.random(n) < 0.05                                  # one row in twenty is scattered at random
xy[noise] = rng.uniform(-4.0, 4.0, (noise.sum(), 2))
pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "ring": ring, "noise": noise}).write_parquet("rings.parquet")

# Fit, and score only the last rows.
mc = rp.spec.micro("mc", features=["x", "y"], half_life=3000.0, eps=0.07)
out = pl.scan_parquet("rings.parquet").rheon.fit_predict([mc]).collect().unnest("mc")
late = out.tail(4000)                                         # once the summaries have formed
# How many clusters, and how each ring and the noise are labeled.
print(f"clusters: {late['n_clusters'][-1]}")
for name, ring_id in (("inner ring", 0), ("outer ring", 1)):
    rows = late.filter(~pl.col("noise") & (pl.col("ring") == ring_id))
    labels = rows.filter(~pl.col("outlier"))["cluster"].n_unique()
    print(f"{name}: {labels} label, {rows['outlier'].mean():.1%} flagged as noise")
print(f"scattered rows: {late.filter(pl.col('noise'))['outlier'].mean():.1%} flagged as noise")
```

```text
clusters: 2
inner ring: 1 label, 0.4% flagged as noise
outer ring: 1 label, 1.5% flagged as noise
scattered rows: 61.1% flagged as noise
```

A scattered row that lands on a ring, or close enough to join one of its
summaries, takes the ring's label, so not every scattered row is flagged.

#### See also

When the number of clusters is known, or a row's cluster depends on the last:

- [`kmeans`](#kmeans--exponentially-weighted-k-means), a fixed number of
  clusters whose centers follow the stream.
- [`hmm`](#hmm--which-regime-are-we-in), Gaussian states whose next state
  depends on the last.

#### Notes

**A summary whose rows stop lingers `half_life · log2(n / (beta_mu · w̄))`
of clock, with `n` the weight it had.** A cluster's label is the smallest id
among its linked summaries, so it lasts as long as that summary does.

**It recovers shapes that `kmeans` cannot.** Measured at 20,000 rows and a
half-life of 3,000, moons, rings and five Gaussians in twenty dimensions all
score an adjusted Rand index (ARI) of 1.000 against the truth. Their `eps`
were 0.07, 0.1 and 0.3, and `kmeans` cannot follow the first two.

**It flags scattered rows and leaves the clusters' rows alone.** On four
clusters with one row in twenty scattered uniformly over a box three times
their extent (40,000 rows, a half-life of 2,000, `eps=0.07`), 94% of the
scattered rows were flagged `outlier`, and 0.3% of the rest. The recipe's
scattered rows read 61% because many land near a ring and take its label.

**It labels a cluster born mid-stream within tens of rows.** On four
clusters, one of which stops halfway through as a fifth is born (a
half-life of 1,000), the new cluster had a label 31 rows in.

### `ew_class` — Gaussian classification on `ew_cov` moments

*API:* [`rheon.spec.ew_class`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ew_class) — *Rust:* [`ewclass.rs`](../crates/rheon-core/src/ewclass.rs) — *Outputs:* [fields](OUTPUTS.md#ew_class)

A Gaussian classifier on exponentially weighted moments, one set per class.

Use `ew_class` to predict a row's class from a column of classes, named in
`label` with its values listed in `classes`. It fits quadratic discriminant
analysis, linear discriminant analysis or Gaussian naive Bayes. Use
[`hmm`](#hmm--which-regime-are-we-in) where no column gives the class and
the model must infer a hidden regime, and
[`kmeans`](#kmeans--exponentially-weighted-k-means) for clusters with no
classes at all.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ew_class),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#ew_class). It takes
`label` in place of `targets`. The parameters every spec takes are in
[What every spec takes](#what-every-spec-takes).

#### The update

Each class `c` keeps one [`ew_cov`](#ew_cov--exponentially-weighted-moments)
state: a weight `n_c`, a mean `μ_c` and a centered covariance `C_c`. `C_c`
is in the population form, with no `ddof` correction, as `ew_cov`'s is under
`bias=True`. `π_c` is a class's share of the weight, and `ω_c` the ridge
`precision_prior`, scaled by `s_c`, which lets a class be predicted from its
first row and fades as `ew_cov`'s prior does. A row is predicted by Bayes'
rule over Gaussian classes, then learned by its own class:

```
π_c = n_c / Σ n         ω_c = precision_prior · s_c
M_c = C_c + ω_c I  ("full")      M = Σ π_c M_c  ("tied")      diag(C_c) + ω_c I  ("diag")
ℓ_c = ln π_c − ½ ln det M_c − ½ (x − μ_c)ᵀ M_c⁻¹ (x − μ_c)
p_c = exp(ℓ_c − max ℓ) / Σ exp(ℓ − max ℓ)                  class = argmax ℓ
n_c ← λ n_c + w·[y = c]        μ_c, C_c ← weighted Welford on the row's own class
s_c ← s_c · λn_c / (λn_c + w)  on the row's own class, and 1 on its first row
```

#### Example

**It predicts a row's class, out-of-sample, about as well as the Bayes rule
that knows the true distributions.** This code makes three Gaussian classes
with their own spreads, and scores the last 5,000 rows, each predicted
before its own class is learned:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# Three Gaussian classes, each with its own spread.
rng = np.random.default_rng(8)
n = 6000
means = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
spread = np.array([[1.0, 0.6], [0.6, 1.0], [0.8, 0.8]])        # each class's own standard deviations
label = rng.integers(0, 3, n)                                 # three classes, equally likely
xy = means[label] + rng.standard_normal((n, 2)) * spread[label]
pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "kind": np.array(["a", "b", "c"])[label]}).write_parquet("classes.parquet")

# Class the last rows, each before its own class is learned.
cl = rp.spec.ew_class("cl", features=["x", "y"], label="kind", classes=["a", "b", "c"],
                      precision_prior=0.1, half_life=float("inf"))
scored = pl.scan_parquet("classes.parquet").rheon.fit_predict([cl]).collect().unnest("cl").tail(5000)
print(f"classed right, out-of-sample: {(scored['class'] == scored['kind']).mean():.3f}")
# The Bayes rule that knows the true distributions, on the same rows.
z = (xy[:, None, :] - means[None]) / spread[None]             # the Bayes rule, from the true means and spreads
best = np.argmax(-0.5 * (z**2).sum(axis=2) - np.log(spread).sum(axis=1), axis=1)
print(f"the Bayes rule, on the same rows: {(best == label)[-5000:].mean():.3f}")
```

```text
classed right, out-of-sample: 0.857
the Bayes rule, on the same rows: 0.860
```

On the last 5,000 rows the model puts 85.7% of the rows in their true
class, against 86.0% for the Bayes rule built from the true means and
spreads.

#### See also

The guides to its settings:

- To choose `covariance_type`, and `window_size` and `embargo` when the
  classes move or arrive late, see [Choose a covariance
  type](how-to/choose-a-covariance-type.md).

For the moments behind each class, or for classes no column names:

- [`ew_cov`](#ew_cov--exponentially-weighted-moments), the moments each
  class keeps, for one set of rows.
- [`hmm`](#hmm--which-regime-are-we-in), the same Gaussian densities for a
  regime that no column names.
- [`kmeans`](#kmeans--exponentially-weighted-k-means), for clusters with no
  classes.

#### Notes

**It reaches the Bayes rate and is calibrated.** The test suite runs it on
200,000 rows of three Gaussian classes in six dimensions, with shares 0.5,
0.3 and 0.2 and each its own covariance. On the second half its accuracy is
within 0.01 of the Bayes rate, and each band of its probabilities is within 0.03 of the
share of rows it classes right.

## Sequential tests and regimes

These models predict no target. Each reads its columns, a target's sign
or its events, and answers, as the rows arrive, a question about the stream: whether
something holds, whether and when it changed, or which regime it is in now.

### `seqtest` — a sequential test of a sign, by betting

*API:* [`rheon.spec.seqtest`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.seqtest) — *Rust:* [`seqtest.rs`](../crates/rheon-core/src/seqtest.rs) — *Outputs:* [fields](OUTPUTS.md#seqtest)

A sequential test of a target's sign by betting, whose evidence can be read
at any row.

Use `seqtest` to read evidence as it accumulates and act the first time it
is enough: whether a column tends to be positive, or whether one spec
predicts closer than another. A p-value cannot be read that way, because
checking it repeatedly inflates its error rate, but the *e-process* this
model keeps can be read at any row. Use
[`rp.eval.seqtest`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.seqtest)
for the same test over a frame in memory, and
[`rp.eval.compare_specs`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.compare_specs)
for several specs' scores in one table.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.seqtest),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#seqtest). The
parameters every spec takes are in [What every spec
takes](#what-every-spec-takes), apart from `weight`, `half_life`
and `lam`, which it refuses. Each row counts as one trial, so to skip a
row, null its target.

#### The update

Per target it keeps the wealth of two gamblers, one betting that the next
sign is positive and one that it is negative. Each stakes the
Krichevsky–Trofimov fraction set by the counts so far, and stakes nothing
unless the counts favor its side. A zero or null target bets nothing and
counts nothing:

```
s = sign(y)                    n⁺, n⁻: the signs counted before this row,  n = n⁺ + n⁻
b⁺ = max(0, (n⁺ − n⁻) / (n + 1))          b⁻ = max(0, (n⁻ − n⁺) / (n + 1))     the stakes
ln E⁺ ← ln E⁺ + ln(1 + b⁺ s)              ln E⁻ ← ln E⁻ + ln(1 − b⁻ s)
```

#### Example

**Its evidence passes the 5% level once a sign holds, and stays below it
when none does.** This code makes a column positive 54% of the time and one
positive half the time, and prints the first row at which each test's
evidence reaches `ln 20`:

```python
import math

import numpy as np
import polars as pl
from rheon import polars as rp

# One column with a slight positive edge, and one with none.
rng = np.random.default_rng(9)
n = 3000
pl.DataFrame({
    "edge": rng.normal(0.1, 1.0, n),                           # positive 54% of the time
    "none": rng.normal(0.0, 1.0, n),                           # positive half the time
}).write_parquet("signs.parquet")

# Test both signs, and find the first row each passes the 5% level.
sign = rp.spec.seqtest("sign", targets=["edge", "none"])
out = pl.scan_parquet("signs.parquet").rheon.fit_predict([sign]).collect().unnest("sign")
for col in ("edge", "none"):
    passed = out.with_row_index("row").filter(pl.col(f"log_e_pos_{col}") >= math.log(20))
    when = f"at row {passed['row'][0]}" if passed.height else "never"
    print(f"{col}: the evidence first passes the 5% level {when}")
```

```text
edge: the evidence first passes the 5% level at row 324
none: the evidence first passes the 5% level never
```

#### See also

The guides to its settings:

- To read its evidence and reject at a level that holds however often you
  look, see [Read a sequential test](how-to/read-a-sequential-test.md).
- To test which of two specs predicts closer, put both in the bank and name
  them in `a` and `b` ([Compare two
  models](how-to/compare-two-models.md#read-the-comparison-on-every-row)).

For the same test outside a bank, or a comparison of many specs at once:

- [`rp.eval.seqtest`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.seqtest),
  the same computation in Polars expressions over a frame in memory.
- [`rp.eval.compare_specs`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.compare_specs),
  several specs' scores in one table.

#### Notes

**Where the `max(0, …)` in a stake never binds, the wealth has the closed
form `2ⁿ B(n⁺+½, n⁻+½) / π`**, the Beta(½, ½) mixture, and a test holds the
bank to it. `rp.eval.seqtest`, the same computation in Polars expressions,
is held to the bank to the bit.

### `corrchange` — has the correlation structure changed?

*API:* [`rheon.spec.corrchange`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.corrchange) — *Rust:* [`corrchange.rs`](../crates/rheon-core/src/corrchange.rs) — *Outputs:* [fields](OUTPUTS.md#corrchange)

Tests of whether the correlations between the feature columns have changed,
by three kinds of test.

Use `corrchange` to test whether the correlations among the columns you
name have changed, with `kind` choosing the test. Use
[`bocpd`](#bocpd--how-long-has-this-regime-lasted) for the age of the
current regime of the features' levels and spreads, and
[`deco`](#deco--one-correlation-for-the-whole-matrix) to follow an
equicorrelation that moves.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.corrchange),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#corrchange). The
parameters every spec takes are in [What every spec
takes](#what-every-spec-takes), apart from `half_life` and `lam`,
which it takes only under `scalar=True`.

#### The update

Each kind answers its own question:

| `kind` | the question | reports | against | a change is found |
|---|---|---|---|---|
| `"monitor"` | was the correlation constant over a span? | on a span's last row | the span's own correlation | at the span's end, up to `window_rows` rows late, and reliably only where it falls well inside a span |
| `"sequential"` | has it left the level a stable history set? | on each row tested against the history, from the fourth on | a history of `window_rows` rows | as soon as it crosses a boundary |
| `"window"` | how big is the change? | on every row once both windows are full, `2·window_rows` rows in | the window before | as soon as the two windows differ |

Under `"monitor"` and `"sequential"`, the statistic is the largest over the
pairs, and the level `alpha` (0.05 by default) is spread over the pairs as
`alpha / npairs` unless `alpha_adjust="none"`.

`kind="monitor"` is the closed-sample constancy test of Wied, Krämer and
Dehling (2012) in its linear form. Its span is `T = window_rows` rows,
centered at their means and scaled by their standard deviations (`x̃`,
`ỹ`), with correlation `ρ̂`. `D̂` is the Bartlett long-run standard
deviation of `ξ`, the delta-method deviation of `ρ̂`. The last row of a
span reports, per pair:

```
ξ_t  = x̃_t ỹ_t − (ρ̂/2)(x̃_t² + ỹ_t²)
Q    = max_{1≤j≤T} |Σ_{t≤j} ξ_t| / (√T·D̂)
crit = q(1 − α) − 0.5826/√T                q the Kolmogorov quantile, 1.3581 at 5%
```

The paper's own statistic, `max_j (j/√T)·|ρ̂_j − ρ̂_T| / D̂`, compares the
correlation of the span's first few rows, `±1` at two rows whatever the data,
with the span's: it flagged 26% of stable spans of 100 rows at a correlation
of 0.9. The partial sums of `ξ`, to which their proof reduces it, form no
such correlation and have the same limit. A pair flags where the same CUSUM
of `x̃² − ỹ²`, which moves only when the two columns' scale ratio moves, does
not pass `crit`; where it does, the flag is withheld, named
`scale_ratio_moved`.

`kind="sequential"` is the monitoring procedure of Wied and Galeano (2013).
A cycle is `window_rows` rows of history, taken as stable, then up to
`monitor_rows` rows, each tested against the history as it arrives. A flag,
or the cycle's last monitored row, ends the cycle, and the row after it
starts a new history. From the history it reads each pair's correlation
`ρ̂_h` and its `D̂`. The `k`-th monitored row then reports, on Fisher's scale,
each side at its own finite-sample variance

```
V_k  = ((k−3)/√(m−3))·|atanh ρ̂_k − atanh ρ̂_h|·(1 − ρ̂_h²) / D̂     m = window_rows
stat = max over pairs of V_k / w((k−3)/(m−3))
w(b) = (1 + b)·(b/(1 + b))^γ               γ = boundary_gamma, 0 ≤ γ ≤ 0.49
```

and flags where `stat` passes `crit`. The paper's own detector, `(k/√m)·|ρ̂_k
− ρ̂_h| / D̂` from the second row, flagged a quarter to a third of the
cycles in their first rows at `γ = 0.45` with nothing changed. This one
holds 0.05 to 0.07 on Gaussian pairs at every `γ`.

Every kind takes its span on the clock too, as `window_size` in place of
`window_rows` (and `monitor_span` in place of `monitor_rows`). Each span,
history or pair of windows is then tested at the rows it holds, and the
window kind's held critical value is rescaled to each report's two windows.

`kind="window"` reports `‖vech(R_pre − R_post)‖₁` (under `norm="linf"`, its
largest absolute entry instead) over two adjacent windows of `window_rows` rows each. `R_pre` and `R_post` are the windows'
correlation matrices, and `vech` stacks their upper triangles into one list.
It flags against a permutation quantile, or against `crit` where one is
given.

The docstring's Notes give each critical value, `D̂` and the dating of a
change, under *The long-run deviation*, *The monitor test*, *The sequential
test* and *The window test*.

#### Example

**The sequential kind flags a correlation break soon after it happens.**
This code makes two columns that are uncorrelated until row 2,200 and
correlate at 0.7 after it, and prints each flag:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# Two columns that start to correlate partway through.
rng = np.random.default_rng(10)
n = 4000
rho = np.where(np.arange(n) < 2200, 0.0, 0.7)                 # the truth: uncorrelated, then 0.7 from row 2200
a, e = rng.standard_normal((2, n))
pl.DataFrame({"a": a, "b": rho * a + np.sqrt(1 - rho**2) * e}).write_parquet("break.parquet")

# The sequential test, then each row it flags.
watch = rp.spec.corrchange(
    "watch", features=["a", "b"],
    kind="sequential",           # each row tested against a stable history as it arrives
    window_rows=500,             # 500 rows of history ...
    monitor_rows=1000,           # ... then up to 1000 rows monitored: a cycle of 1500 rows, the second from row 1500
)
out = pl.scan_parquet("break.parquet").rheon.fit_predict([watch]).collect().unnest("watch")
for r in out.with_row_index("row").filter(pl.col("flag")).iter_rows(named=True):
    print(f"flag at row {r['row']}: stat {r['stat']:.2f} past crit {r['crit']:.2f}")
```

```text
flag at row 2284: stat 1.83 past crit 1.83
```

The break falls in the second cycle's monitored rows, which start at row
2,000, and the flag comes 84 rows after it. No row of the first cycle, or of
the second cycle before the break, is flagged.

#### See also

The guides to its settings:

- To catch an early change sooner with `boundary_gamma`, the `γ` of `w(b)`,
  and to count a run of the window kind's flags as one change, see [Detect a
  break](how-to/detect-a-break.md#watch-the-features-correlations-with-corrchange).

For the age of a regime, or a correlation that moves rather than breaks:

- [`bocpd`](#bocpd--how-long-has-this-regime-lasted), a posterior over the
  age of the current regime.
- [`deco`](#deco--one-correlation-for-the-whole-matrix), an
  equicorrelation that moves with the stream.
- [REGIMES.md](REGIMES.md), the size and power of these tests,
  measured against their papers.

#### Notes

**The sequential kind's critical value is solved, not simulated.** Where
the paper simulates its quantile `q`, above `γ = 0`, this library solves it
as a diffusion with an absorbing boundary
([`boundary.rs`](../crates/rheon-core/src/boundary.rs)). The paper allows
`γ` up to 1/2, but the solve's work grows as `1/(1/2 − γ)`: seconds at 0.49
and hours just under 1/2. So `boundary_gamma` above 0.49 is refused.

**A window's report takes the same work whatever the window's length.** Each window's
correlations are merged from summaries of blocks of rows: a block's count,
mean and centered co-moments, kept once and combined by the pairwise update
rule. So for `p` features a report reads only the rows at each window's
ends, about `p/2 + 3` of them, plus a handful of merges. A row that leaves a
window is never subtracted, so nothing drifts. Each block keeps its mean as
an offset from one of its own rows, so a level of 1e8 under a spread of 1 costs no accuracy: the
statistic is within a few units of rounding of the exact one.

**Its redraws are held for a window's length.** A redraw shuffles the
`2 * window_rows` pooled rows `n_perm` times, so `permute_every_rows`
defaults to `window_rows`. The redraws then cost `n_perm * p**2` a row
whatever the window. Each value is held for `permute_every_rows` reports, so
a report is compared with draws from windows up to
`permute_every_rows − 1` rows older than its own.

**The monitor's size is at or under the nominal 5% at every span length and
correlation,** where the paper's statistic was not ([Is the monitor the size
its paper says?](REGIMES.md#2-is-the-monitor-the-size-its-paper-says) and
[And is it the power its paper says?](REGIMES.md#3-and-is-it-the-power-its-paper-says)). The sequential
detector's size on Wied and Galeano's own design is 0.05 to 0.07 at every
`boundary_gamma`, where their Table 2 reads up to 0.174 at 0.45 ([The
sequential detector against its
paper](REGIMES.md#9-the-sequential-detector-against-its-paper)). Its critical
values are within 0.03 of their Table 1. The window kind flags `alpha` of its
reports, and a stretch of a window's length of reports holds a flag about a
third of the time, as the docstring's Notes give under *The window test*.

### `bocpd` — how long has this regime lasted?

*API:* [`rheon.spec.bocpd`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.bocpd) — *Rust:* [`bocpd.rs`](../crates/rheon-core/src/bocpd.rs) — *Outputs:* [fields](OUTPUTS.md#bocpd)

Bayesian online changepoint detection, a posterior over how long the current
regime has lasted.

Use `bocpd` (Adams & MacKay 2007) to know how long the current regime has
lasted. It keeps a probability distribution over the **run length**, the
rows since the last break, so it can say "we are forty rows into a regime",
not only "something broke". Use
[`corrchange`](#corrchange--has-the-correlation-structure-changed) for a
test of a change in correlation, and a regression's `emit_breaks` for a
break in its coefficients.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.bocpd),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#bocpd). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes),
apart from `half_life` and `lam`, which it refuses: the run-length posterior
does the forgetting, and the hazard, the expected run length, sets how fast:
`hazard_rows` in rows, or `hazard` in clock time.

#### The update

With `H = 1/hazard_rows` the chance of a break after each row, and `π_r` the
posterior predictive of run length `r` for this row, the model runs their
Algorithm 1:

```
growth:      P(r_t = r+1, x_{1:t}) = P(r_{t−1} = r, x_{1:t−1})·π_r·(1 − H)
changepoint: P(r_t = 0,   x_{1:t}) = Σ_r P(r_{t−1} = r, x_{1:t−1})·π_r·H
```

Under `hazard`, `τ` is the expected clock time between changepoints: a
number of the clock's units, rows without a clock, or a duration such as
`"1h"` on a temporal clock. The step of `d` into a row carries the chance
`h = 1 − exp(−d/τ)` that a break fell inside it. That chance belongs before
the row, so `h` of every run's mass moves to the empty run before the row
is read. Algorithm 1 then runs with `H = 0`, since no time has passed since
the row.

`hazard_rows` is Adams and MacKay's hazard, one step an observation: a break
is as likely after any row, however far apart the rows fall. `hazard` says
breaks come in time, so a long gap makes one near certain and a row that
shares the last row's stamp cannot begin a run. On steps of one unit the
two are one model at `hazard_rows = 1/(1 − e^{−1/τ})`, every output the
same but `p_change`, which the per-row form reads with `H` at `r = 0`:
`p_rows = H + (1 − H)·p_clock`.

#### Example

**`run_mode` dates a break, while `p_change` may hardly move.** This code
makes a column whose spread quadruples at row 1,000, and reads the row each
row's current run began on, `row - run_mode`:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# A column whose spread quadruples partway through.
rng = np.random.default_rng(11)
n = 2000
scale = np.where(np.arange(n) < 1000, 1.0, 4.0)               # the truth: the spread quadruples at row 1000
pl.DataFrame({"x": rng.standard_normal(n) * scale}).write_parquet("spread.parquet")

# Fit, and date each row's current run from run_mode.
runs = rp.spec.bocpd("runs", features=["x"], hazard_rows=500.0)    # a break expected every 500 rows
out = pl.scan_parquet("spread.parquet").rheon.fit_predict([runs]).collect().unnest("runs")
out = out.with_row_index("row").with_columns(began=pl.col("row") - pl.col("run_mode"))   # the row the current run began on
# The dating of the break, and how little p_change moved.
found = out.filter((pl.col("row") >= 1000) & (pl.col("began") >= 1000)).row(0, named=True)
print(f"first row whose run began after the break: {found['row']}, dated to row {found['began']}")
for row in (1050, 1999):
    print(f"row {row}: the regime began at row {out['began'][row]}")
print(f"largest p_change on rows 1000 to 1049: {out['p_change'][1000:1050].max():.2f}")
```

```text
first row whose run began after the break: 1009, dated to row 1001
row 1050: the regime began at row 1001
row 1999: the regime began at row 1001
largest p_change on rows 1000 to 1049: 0.02
```

Nine rows after the break the posterior has accepted it, and dates it within
a row of the truth. No single row of the wider regime is surprising enough
to lift `p_change`, which is why the answer is read from `run_mode`.

#### See also

The guides to its settings:

- To choose `hazard`, the priors and `robust_beta`, see [Choose a
  changepoint hazard](how-to/choose-a-changepoint-hazard.md).
- To read the regime's age from `run_mode` and use `p_change` only as an
  alarm, see [Detect a break](how-to/detect-a-break.md#date-a-regime-with-bocpd).

For a test of a change in correlation, or a fixed set of regimes:

- [`corrchange`](#corrchange--has-the-correlation-structure-changed),
  tests for a change in the correlations among the features.
- [`hmm`](#hmm--which-regime-are-we-in), a fixed number of regimes the
  stream moves between.
- [Two changepoint detectors on the same break](REGIMES.md#5-two-changepoint-detectors-on-the-same-break),
  `bocpd` beside `corrchange` on the same break.

#### Notes

**The pruned posterior is held to the whole one.** At the defaults,
`prune_below=1e-6` and `max_run=10,000`, the run-length posterior is held
to the one the `bayesian_changepoint_detection` package computes with
nothing pruned. Its `p_change` is held to the package's within a bound set
by the mass pruned.

**Two features or more add their log densities in blocks.** Under
`covariance_type="diag"`, a row's log predictive is a sum over the features
of a Student's t log density. Two of its terms are a logarithm of the
feature's own value: `ln ψₙ` of the scale, and `ln(1 + z²/νₙ)` of the
squared standardized distance. Those logarithms are most of the work of a
row. For two features or more, each term's logarithm is taken once for a
block of 8 features, as the log of the block's product. The term that does
not depend on the data, `ln(π(κₙ+1)/κₙ)`, is held once for each run length.
A block whose product would leave the range of normal doubles takes its
logarithms one at a time. The result differs from adding the features'
logarithms one at a time by rounding alone, within `8·2⁻⁵²·(1 + Σ|terms|)`
of the log density. That shows in the last digits of `p_change`, `loglik`
and the predictions, which are functions of the log posterior. One feature
takes a single logarithm of each, and `covariance_type="full"` is not
affected.

**It flags a step up in the spread at once, and dates a step down by the
run length.** Adams and MacKay's own finance example (a = 1, b = 1e-4,
hazard = 250) is `degrees_of_freedom_prior=2.0`, `covariance_prior=[2e-4]`
and `hazard_rows=250.0`. Under it, with `max_run=600`, a tenfold step up in the
spread lifts `p_change` above 0.5 within ten rows, and `run_mode` dates it
to its row. The step back down is dated within one row.

### `hmm` — which regime are we in

*API:* [`rheon.spec.hmm`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.hmm) — *Rust:* [`hmm.rs`](../crates/rheon-core/src/hmm.rs) — *Outputs:* [fields](OUTPUTS.md#hmm)

A Gaussian hidden Markov model of the features, filtered one row at a time.

Use `hmm` to ask "which regime are we in", not only "which regime does this
row look like": a transition matrix carries what the rows before have shown
into each row's answer.
[`ew_class`](#ew_class--gaussian-classification-on-ew_cov-moments)
classifies a row against *labeled* Gaussians, so use it when a column names
each row's class. An `hmm` does the same arithmetic without labels, and its
regime is a *hidden state*.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.hmm),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#hmm). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes).

#### The update

Here a hidden state means one of the `n_components` regimes, not the bank's
saved state. The model runs Hamilton's filter one row at a time, from the
filtered `p` the previous row left. Every output but `coef` is read before
the row is learned. With `μ_l` and `Σ_l` state `l`'s mean and the
covariance the row reads, and `Π` the transition matrix:

```
p1_l   = Σ_k p_k·Π_kl                      the predicted state
f_l    = N(x | μ_l, Σ_l)                   the state's density
loglik = ln Σ_l p1_l·f_l                   the row's log-likelihood
p_l   ← p1_l·f_l / Σ                       the filtered state
```

A row splits across the states at weight `w·p_l`, which sums to `w`, so
`weight_sum` follows the same recursion as in every model. The transition
matrix is learned from the filtered joint of consecutive states. `N_kl` are
its decayed counts. `τ_kl` is the Dirichlet pseudo-count:
`transition_prior`, or `transition_prior · n_components · transition_kl`
where `transition` is given, so that the given matrix is the prior's mean.
It keeps the matrix row of a state never visited a distribution:

```
N_kl ← λ N_kl                                    before the row reads Π
Π_kl = (N_kl + τ_kl) / Σ_l (N_kl + τ_kl)
ξ_kl = p_k(t−1)·Π_kl·f_l / Σ                     N_kl ← N_kl + w·ξ_kl
```

Each state keeps its weight `W_l`, its means and its mean-form co-moments
`M_l`, which take the row as `ew_cov`'s do. With `δ` the row's deviation
from the state's mean before the row, `a = W_l / (W_l + w·p_l)` and
`b = w·p_l / (W_l + w·p_l)`:

```
M_l ← a·M_l + a·b·δδᵀ                            W_l ← W_l + w·p_l
```

**Under `"tied"`, and with `learn = False`,** `Σ_l = M_l + ω_l·I`, with
`ω_l` the `precision_prior` ridge as it has decayed with the state's data.

**Under `"full"` and `"diag"` with `learn = True`, each state's covariance
forgets toward a prior** (stabilized forgetting: Kulhavý and Zarrop, "On a
general concept of forgetting", Int. J. Control 58(4), 1993). Exponential
forgetting flattens a state's posterior toward nothing: a state that wins no
rows for a while keeps a ridge that decays with its weight, and on its first
row back its matrix collapses onto that row, so a returning regime is not
recognized. Stabilized forgetting flattens it toward a prior worth
`P₀ = prior_rows` rows (the number of features by default), and a prior
worth `P₀` rows stays worth `P₀` rows. With `ψ` the features' pooled
within-state variance:

```
W_l ← λ·W_l                                      the row's decay, before the read
Σ_l = (W_l·M_l + P₀·diag(ψ)) / (W_l + P₀)
```

With no data (`W_l → 0`) a state reads `diag(ψ)`, a broad density, never a
collapsed one. With no decay it is scikit-learn QDA's `reg_param`
shrinkage, `reg_param = P₀ / (W_l + P₀)`, toward `diag(ψ)`. `ψ` follows
each row's squared deviations from the states' means before it, weighted
by the filtered state, at ten times the half-life, from the seeded states'
variances pooled by weight:

```
e_i  = Σ_l p_l·(x_i − μ_li)²
V   ← λ^(1/10)·V + w
s_i ← s_i + (w/V)·(e_i − s_i)
ψ_i  = max(s_i, 2⁻¹⁰⁰·max_j s_j)                 precision_prior where every s_i is 0
```

**Where the states are given (`means` and `covs`), each state's prior is
the covariance it was given.** With `G_l` state `l`'s given matrix:

```
Σ_l  = (W_l·M_l + P₀·diag(√ψ)·G_l·diag(√ψ)) / (W_l + P₀)
e_i  = Σ_l p_l·(x_i − μ_li)² / G_l,ii             in units of the given variances
```

`s` follows `e` as above and starts at 1. So a state reads `G_l` on the
first row, and for as long as the rows have the given variances, whatever
share of the rows each state holds. `ψ` rescales each feature by how far
the rows' variances have moved from the given ones, so given covariances
off by a factor, or a stream whose scale drifts, are followed. A state with
no data reads the regime it was given, at the scale the rows have now.

The diagonal prior reads one matrix for every state. Under it, regimes that
differ in correlation or in variance alone are told apart by their data
alone, and a state absent for a while reads as the others do. Measured on
streams of two zero-mean regimes at correlation 0.1 and 0.8, the states
given at the truth: a learning filter put 0.79 of the rows in the true
state under the diagonal prior, and 0.98 under each state's own
([REGIMES.md](REGIMES.md#1-does-hmm-recover-the-stream-that-made-it)).

What the given covariances say about each regime's shape is trusted at
`prior_rows` rows for as long as the model runs. So given covariances of
the wrong shape cost. With the two regimes' matrices given the wrong way
round, a regime back from an absence kept the label it was given in 14 of
24 simulated returns, against 24 under the diagonal prior. A steady
stream's posteriors were 9% further from the exact filter's. Given
covariances that are the same multiple of the identity for every state are
the diagonal prior exactly. The same diagonal with variances that differ
keeps its shape: as plain variances, the floor on `psi` (`2^-100` of the
largest) would lift a variance more than `2^100` below the largest, which
in units of the given variances reads as given.

The study behind it simulated two-state streams and scored each against
the exact forward pass with the true parameters (hmmlearn). A regime absent
for 20 to 500 half-lives and then back was recovered in 136 of 144 `full`
runs under the prior, and in 60 under the decaying ridge; under `"diag"`,
in 136 and 108. Steady and drifting streams were as well fitted or better.

A state's density reads `Σ_l` through its Cholesky factor. Under
`covariance_type = "full"` with 13 features or more, each state keeps that
factor itself, as `s·L·Lᵀ` with a scalar `s` between 1/4 and 1. The row
moves it by plane rotations in time `d²`. Narrower states keep their
co-moments and factorize them in time `d³`. The prior's share of `Σ_l`
changes with `W_l`, a diagonal of full rank that no rotation adds. So the
factor holds the prior as factored, `f`, which decays with the data. One
factorization of each state moves it back to its target whenever any
coordinate has drifted more than 1/16 from it. With `W'` the weight after
the row:

```
Σ_l    = s·L·Lᵀ = (W_l·M_l + diag(f)) / (W_l + P₀)
f_i   ← λ·f_i                                     every row, with W_l
f_i   ← (1 + 1/32)·P₀·ψ_i                         when any f_i leaves [15/16, 17/16]·P₀·ψ_i
s'     = s·(W_l + P₀) / (W' + P₀)                 the row learned
L'L'ᵀ  = L·Lᵀ + (a·w·p_l) / ((W' + P₀)·s')·δδᵀ    by plane rotations
```

So the factor form reads `Σ_l` within 1/16 of the prior's share of it: a
quadratic form within a factor 16/15 of the exact recursion's. Over 2,000 rows
(13 and 20 features, half-lives of 20 and 500), filtered posteriors parted from
the exact recursion by at most 0.009. Early in a stream, where the prior carries
more weight (200 rows at a half-life of 30 across 13 to 20 features), they
parted by up to 0.050. Every coordinate decays alike, so all fall due together,
about every `1.5·(1/16)·h / ln 2` rows at a half-life of `h` rows (7 rows at 50,
68 at 500). Measured on arm64 with two states, against the decaying ridge it
replaces, a row of the factor form costs 0.95 to 0.98 of its time at 13 to
32 features and a half-life of 500 rows. At a half-life of 50 rows it costs
1.1 to 1.2. A `"diag"` row costs 1.19 to 1.29 of its time, and a `"full"` row
below 13 features 1.08 to 1.16. Under given states' own matrices the
factor holds `diag(√f)·G_l·diag(√f)` in place of `diag(f)`, and a row costs
1.00 to 1.05 of the diagonal prior's time at 4 to 32 features, in every
form.

#### Example

**Because regimes last, the filter beats a rule that knows the true
centers.** This code makes two overlapping blobs, 1.5 apart in each
coordinate, and a sticky chain that switches between them about once in 200
rows. It compares the filter's state with the nearest true center, row by
row:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# A sticky two-state chain, and two overlapping blobs it switches between.
rng = np.random.default_rng(12)
n = 6000
state = np.zeros(n, dtype=int)
for i in range(1, n):                                         # a sticky chain: each row has a 0.5% chance of a switch
    state[i] = 1 - state[i - 1] if rng.random() < 0.005 else state[i - 1]
centers = np.array([[-0.75, -0.75], [0.75, 0.75]])            # two blobs of unit spread
xy = centers[state] + rng.standard_normal((n, 2))
pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "truth": state}).write_parquet("states.parquet")

# Filter the states, then score the last rows against the nearest true center.
hidden = rp.spec.hmm("hidden", features=["x", "y"], n_components=2, precision_prior=1e-2,
                     warm_rows=500, half_life=float("inf"))
rows = pl.scan_parquet("states.parquet").rheon.fit_predict([hidden]).collect().unnest("hidden").tail(5000)
right = (rows["state"] == rows["truth"]).mean()               # the states are numbered as seeded, so read either way
nearest = ((rows["x"] + rows["y"] > 0).cast(pl.Int64) == rows["truth"]).mean()   # the nearest true center
print(f"rows in their true state: the filter {max(right, 1 - right):.2f}, the nearest true center {nearest:.2f}")
```

```text
rows in their true state: the filter 0.92, the nearest true center 0.85
```

The nearest center reads each row alone. The filter carries what the rows
before told it through the transition matrix it learns, so a row that lands
nearer the other blob does not flip its state.

#### See also

The guides to its settings:

- To give the transition matrix a sticky prior, and keep a single extreme
  row from capturing a state, see [Keep `hmm` states on their
  regimes](how-to/keep-hmm-states-on-their-regimes.md).

For labeled classes, or for the age of a regime rather than its name:

- [`ew_class`](#ew_class--gaussian-classification-on-ew_cov-moments), the
  same Gaussian densities against labeled classes.
- [`kmeans`](#kmeans--exponentially-weighted-k-means), whose `init` seeds
  the states.
- [`bocpd`](#bocpd--how-long-has-this-regime-lasted), how long the current
  regime has lasted.

#### Notes

**Its learning is held to the rule rebuilt with hmmlearn and scipy.** Each
row's filtered posterior comes from hmmlearn's `GaussianHMM`, and each joint
of consecutive states from scipy's Gaussian densities, with none of the
model's arithmetic. Its seeding is held to scikit-learn's, as `kmeans`'
is. [Does `hmm` recover the stream that made
it?](REGIMES.md#1-does-hmm-recover-the-stream-that-made-it)
measures how often it puts a row in the state that generated it, seeded from
the rows and given the true covariances.

### `hawkes` — how much does each event excite the next?

*API:* [`rheon.spec.hawkes`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.hawkes) — *Rust:* [`hawkes.rs`](../crates/rheon-core/src/hawkes.rs) — *Outputs:* [fields](OUTPUTS.md#hawkes)

A self-exciting point process (Hawkes): a base rate for each node, and how
many events of each node one event of each node causes.

Use `hawkes` on a stream of events that come in bursts, where each event
makes the next likelier for a while: trades, messages, failures. Each
target is a node, and a value above 0 on a row is an event of that node, of
that mark (1 for an event, a count for several at once). Each row reports
each node's intensity, its expected events per unit of the clock, read just
before the row's events from the fit on the rows before them.

Its parameters, their defaults and what each refuses are in its
[docstring](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.hawkes),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#hawkes). The parameters
every spec takes are in [What every spec takes](#what-every-spec-takes),
apart from `features`, which it refuses. A row's weight only decides whether
its events are learned: 0 skips them, any weight above 0 learns them. A row
of weight 0 also counts in no clock: the next row steps as if it were not
there, it is in no window, and its weight is in no `weight_sum`; it is still
read, from the last row that counted.

#### The update

With `R_ju` node `j`'s excitation at the kernel decay `β_u`, `φ = (1, R)`,
the intensity `λ_i = θ_iᵀ φ`, and `f = λ` the row's decay over the clock
step `d` (`γ = ln 2 / half_life`):

```
R_ju ← e^(−β_u d) R_ju      b ← f b      Q ← f Q      G₀₀ ← f G₀₀ + (1 − f) / γ
each node j with an event of mark m_j, from R before the row's events:
b_j ← b_j + m_j φ           Q_jl ← Q_jl + m_j m_l       R_ju ← R_ju + m_j β_u
G₀ₐ = (β_a b_j(a)[0] − R_a) / (β_a − γ)
Gₐᵦ = (β_a b_j(a)[b] + β_b b_j(b)[a] + β_a β_b Q_j(a)j(b) − R_a R_b) / (β_a + β_b − γ)
θ_i = G⁻¹ b_i
```

`θ_i` minimizes the least-squares contrast `Σ_i ∫ λ_i² dt − 2 ∫ λ_i dN_i`,
each instant weighted by its decay. `G` is not kept, since it can be read
from `b` and `Q`. The state is `8 (k + D (1 + k) + D (D + 1) / 2 + 2)` bytes
for `D` nodes and `k = D U` excitations: 64 bytes for one node and two
decays, 200 for two nodes and three.

#### The kernels, and a temporal clock

The kernels are given by their decays, `kernel_decays=[1.0, 5.0]`, rates per
unit of the clock, or by their half-lives, `kernel_half_lives=[0.5, 2.0]`,
with `β = ln 2 / half-life` (give one of the two). A rate has no duration
form, so on a temporal clock the kernels are given by their half-lives as
durations, as `half_life` is: `kernel_half_lives=["500ms", "2s"]`,
`window_size="5m"`, `gap_cap="1h"`. A temporal clock is read in its own
integer nanoseconds, a duration is its seconds, and the model then runs in
seconds. Take the same spec on a numeric clock of seconds, with the decays `ln
2 / half-life` converted by hand. Its output is the same to the bit where a
step's seconds are the same double on both clocks (quarter seconds, say), and
within 1e-13 where they differ in the last bit of a step. **Quantities measured on a
temporal clock reach the output in seconds**, as for `holt`'s trend: each
node's `intensity` and the base rate `mu` in `coef` are events per second,
and each `alpha`, a count of events, has no unit. A term's name carries the
kernel as it was given: `alpha_<j>_beta2` for a decay, `alpha_<j>_h500ms`
for a half-life.

#### Example

**It recovers the parameters of a process it did not see made.** This code
simulates one node from a base rate of 0.5 and a kernel of `α = 0.6` at
`β = 2`, by Ogata's thinning, and prints the fit:

```python
import numpy as np
import polars as pl
from rheon import polars as rp

# Simulate one node by thinning: propose from an upper bound, then accept.
rng = np.random.default_rng(3)
t, excite, times = 0.0, 0.0, []
while t < 5_000.0:
    # Propose the next time from the current intensity, which only decays.
    bound = 0.5 + excite
    step = rng.exponential(1 / bound)
    excite *= np.exp(-2.0 * step)
    t += step
    # Accept it with the ratio of the intensity there to the bound.
    if rng.uniform() * bound <= 0.5 + excite:
        times.append(t)
        excite += 0.6 * 2.0
pl.DataFrame({"t": times, "event": 1.0}).write_parquet("events.parquet")

# Fit the kernel at its true decay, with no forgetting.
model = rp.spec.hawkes(
    "h", targets=["event"], kernel_decays=[2.0], clock="t", half_life=float("inf"), gap_cap=1e9
)
bank = rp.ModelBank([model])
bank.fit(pl.scan_parquet("events.parquet"))

# Read the fit's terms by name and compare them with the truth.
coef = {r["term"]: r["coef"] for r in bank.coef().iter_rows(named=True)}
print(f"mu: true 0.50, found {coef['mu']:.2f}")
print(f"alpha: true 0.60, found {coef['alpha_event_beta2']:.2f}")
```

```text
mu: true 0.50, found 0.51
alpha: true 0.60, found 0.59
```

#### See also

For the age of a burst rather than its intensity, or the evidence that one
spec's events come faster than another's:

- [`bocpd`](#bocpd--how-long-has-this-regime-lasted), how long the current
  regime has lasted.
- [`seqtest`](#seqtest--a-sequential-test-of-a-sign-by-betting), evidence
  you can read at any row.

#### Notes

**Its fit is tick's least-squares minimiser.** tick's
`ModelHawkesSumExpKernLeastSq` computes the same contrast, divided by the
number of events. The fit agrees with its minimiser to 4e-14 of the largest
coefficient, over one to three nodes and one to three decays. A row's
intensity agrees with tick refit on the rows before it, and a window's fit
with tick refit on the window's events. The forgotten fit, which tick does
not compute, is held to the contrast written out as sums over pairs of
events.

**A fit the events cannot identify is withheld.** Two nodes whose events
fall on the same rows give `G` a condition number near `1e14`, and an exact
solution of digits of rounding. Where a pivot of `G` scaled to a unit
diagonal falls below `1e-10`, the row reports no intensity. So it is where
a node has no excitation over the observation: before its first event, or
under a window that holds no event of it before the row. Its column of `G`
is then 0, and its kernels' coefficients mean nothing. Under a window this
is decided from the count of each node's events the window holds, not from
`G`. Taking a node's events back leaves rounding in its sums, which the
scaling to a unit diagonal would pass as a fit, so its sums are set to 0
as its last event leaves. `withheld_reason` names such a row
`not_identified`.

**The window keeps its rows, counted in events or in clock units.**
`window_events = W` fits the last `W` rows that carried an event, alone.
`window_size = D` fits the events of the last `D` clock units, alone, and
observes the interval `(T − D, T]` (from the first row, until there are `D`
units of it), so a stretch with no event in it is observed as one. Either
way dropping a row takes back exactly what it added, through the sums of its
excitation of the later rows, which are kept in two blocks so that no sum is
rescaled by a growing exponential. No row's time is held as the clock's
sum: a row holds its gap to the row before it, so a gap of 0.25 after a
quiet stretch of `1e16` is kept, where `1e16 + 0.25` would round to `1e16`. A fit with `window_size` agrees with
tick refit on the events inside the span to 1e-14 of the largest
coefficient. Give one of the two: they are two windows, and the fit holds
one. The span is a number of the clock's units, or of rows where the spec
has no clock, or a duration on a temporal clock.

**The window's edge.** A row leaves a `window_size` window at the first row
that finds it `D` old, before that row's intensity is read, decided on the
rows' exact stamps and not on a clock summed in doubles. The edge is that of
the window operators and Polars' `rolling_*_by(closed=)`. `closed = "right"`
(the default) keeps the rows less than `D` old, so a row exactly `D` old has
left and the window is `(T − D, T]`. `closed = "both"` keeps it, `[T − D,
T]`. `"left"` and `"none"` would leave out the row the window ends
at, which a model cannot do, and are refused.

**The weight under a window.** `weight_sum`, and the `min_weight` gate that
reads it, count the weight of the rows with an event that the window holds,
as the last row left them. A row that leaves at this row still counts in
what the row reports, as weight a decay takes at this row is still
undecayed in it. A row with no event is no part of the
fit, and the model keeps no record of it, so its weight is not counted. Set
`min_weight` at most the weight a window can hold.

**The state's bytes are held to `max_state_bytes`.** The window holds `8 (2 + D
+ k)` bytes a row (its clock, its weight, its marks and its `k` sums), 40 for
one node and two decays, and `window_events` bounds its rows, so a group's
whole state is known when the spec is built. A spec past `max_state_bytes` is
refused, and one past half of it is warned of. `max_state_bytes` may be at
most 64 MiB a group, as every model's. At its default of 256 KB, that
is about 6,551 rows of one node and two decays, or 3,273 of two nodes and
three. A `window_size` window holds as many events as its span does, which
the data decides, so each row holds a stamp of 32 bytes more. The bytes are
checked as the rows go in. The first row to take a group past half of
`max_state_bytes` gives a warning, once for the group, naming the group and
the bytes. The row that would take it past the cap is not learned, and the
bank stops with an error naming the group and the bytes, having learned part
of the chunk. A spec whose state without any event is already past the cap
is refused when it is built.
