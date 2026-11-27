<!-- kind: map -->
# Every feature

This page lists every feature of rheon, one line each, with a link
to the page that shows it, for a reader who wants the whole list or is
looking for a capability by name. A name links its entry in the [API
reference](https://rheon-dev.github.io/rheon/), which gives its
parameters and their defaults, and the last column links the guide that
shows it in use. `rheon` is the package, `import rheon`, and `rp` its Polars
interface, `from rheon import polars as rp`. `lf.rheon`
stands for the `.rheon` methods of a `LazyFrame` and of a `DataFrame`
alike. The [README's map](../README.md#where-to-go-next) groups the same
pages by the task you have in hand.

| family | its features |
|---|---|
| [Models](#models) | the forty-four models a spec can name |
| [Running a bank](#running-a-bank) | a bank in a query, in a Python loop, and its output as Arrow |
| [Saving, loading and serving](#saving-loading-and-serving) | a bank's state, written and read back |
| [Reading a bank](#reading-a-bank) | what a bank holds, and the names of its output |
| [Preparing a stream](#preparing-a-stream) | features and targets made from windows over the stream, and a refresh grid for series that tick at their own times |
| [Scoring and comparing](#scoring-and-comparing) | `rp.eval`: scores, tests and searches over a bank's output |
| [Fits from Grams](#fits-from-grams) | `rheon.gram`: fits over any blocks and columns, from the Grams one pass leaves |
| [Correlation matrices](#correlation-matrices) | `rheon.corr`: repair, shrink and score a correlation matrix |
| [Simulation](#simulation) | `rp.sim` |
| [Parameters every spec takes](#parameters-every-spec-takes) | the parameters shared by every model, by family |
| [Errors and warnings](#errors-and-warnings) | the classes a caller can catch or filter |
| [Versions and the environment](#versions-and-the-environment) | the version numbers, and the environment variables the library reads |
| [The command line](#the-command-line) | `rheon`, its flags and its configuration file's keys |

## Models

Each model is built by a function of `rheon.spec`, whose first parameter is
the spec's name. [The models at a glance](MODELS.md#the-models-at-a-glance)
compares them, and [Choose a model](how-to/choose-a-model.md) says which
to start with.

| model | what it is | its pages |
|---|---|---|
| [`rheon.spec.ewridge`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ewridge) | exponentially weighted ridge regression on running sums, a grid of ridges and feature sets solved from one set of sums | [entry](MODELS.md#ewridge--exponentially-weighted-ridge-regression) · [choose a ridge and a half-life](how-to/choose-a-ridge-and-half-life.md) |
| [`rheon.spec.rls`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.rls) | recursive least squares, which solves the ridge fit exactly on every row | [entry](MODELS.md#rls--recursive-least-squares) · [targets that go missing on different rows](how-to/fit-targets-that-go-missing-on-different-rows.md) |
| [`rheon.spec.lasso`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.lasso) | a lasso or elastic-net path, with the penalty chosen as the stream runs | [entry](MODELS.md#lasso--lasso-path-its-penalty-chosen-as-it-runs) · [choose columns on the lasso path](how-to/choose-columns-on-the-lasso-path.md) |
| [`rheon.spec.kalman`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.kalman) | a regression whose coefficients drift, tracked by a Kalman filter | [entry](MODELS.md#kalman--random-walk-β-dynamic-linear-model) · [choose a drift speed](how-to/choose-a-kalman-drift-speed.md) |
| [`rheon.spec.huber`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.huber) | Huber regression, which down-weights the rows a prediction misses by far | [entry](MODELS.md#huber--robust-regression) · [the robust scale](explanation/robust-scale.md) |
| [`rheon.spec.quantile`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.quantile) | quantile regression, the fit that a share `tau` of the targets fall below | [entry](MODELS.md#quantile--robust-regression) |
| [`rheon.spec.expectile`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.expectile) | expectile regression, the squared error weighed `tau` above the fit and `1 - tau` below it, in one pass or exact over a window | [entry](MODELS.md#expectile--expectile-regression) |
| [`rheon.spec.sgd`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.sgd) | stochastic gradient descent under squared, Huber, quantile, ε-insensitive, Poisson and logistic losses | [entry](MODELS.md#sgd--stochastic-gradient-descent) · [choose a learning rate](how-to/choose-a-learning-rate.md) · [a count target](how-to/fit-a-count-target.md) · [bound the coefficients](how-to/bound-the-coefficients.md) |
| [`rheon.spec.pa`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.pa) | passive-aggressive regression, the smallest step that brings each row within a margin | [entry](MODELS.md#pa--passive-aggressive-regression) · [follow a level that moves](how-to/follow-a-level-that-moves.md) |
| [`rheon.spec.ftrl`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ftrl) | FTRL-proximal regression, logistic for a 0/1 target, with an L1 penalty that zeroes coefficients | [entry](MODELS.md#ftrl--online-logistic-regression) · [choose its penalties](how-to/choose-ftrl-penalties.md) |
| [`rheon.spec.glm`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.glm) | Poisson, Gamma, binomial and Tweedie regression, a Newton step a row on the likelihood | [entry](MODELS.md#glm--generalized-linear-models-poisson-gamma-binomial-tweedie) · [a count target](how-to/fit-a-count-target.md) |
| [`rheon.spec.holt`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.holt) | Holt's linear trend, a baseline that extrapolates the target's own level and slope | [entry](MODELS.md#holt--holts-linear-trend) · [choose Holt's smoothing](how-to/choose-holts-smoothing.md) |
| [`rheon.spec.ar`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ar) | an autoregression for each target, AR(p), on lags the model keeps itself, with exogenous features | [entry](MODELS.md#ar--an-autoregression-on-the-targets-own-past) |
| [`rheon.spec.isotonic`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.isotonic) | isotonic regression of each target on one feature, the monotone calibrator, over every row, a window or a decay; scikit-learn's fit | [entry](MODELS.md#isotonic--a-monotone-function-of-one-feature) |
| [`rheon.spec.krls`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.krls) | kernel ridge regression, RBF, Laplacian, polynomial or linear, with ALD sparsification or an exact sliding window | [entry](MODELS.md#krls--online-kernel-regression) |
| [`rheon.spec.vector_ar`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.vector_ar) | a vector autoregression of the targets together, VAR(p), with Granger tests on every row and its impulse responses, variance decomposition and roots read with `ModelBank.vector_ar` | [entry](MODELS.md#vector_ar--a-vector-autoregression-with-granger-tests) |
| [`rheon.spec.hoeffding_tree_regressor`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.hoeffding_tree_regressor) | an incremental regression tree, one per target, splitting by variance reduction under a Hoeffding bound within a per-group byte cap | [entry](MODELS.md#hoeffding_tree_regressor--incremental-regression-tree) |
| [`rheon.spec.ew_cov`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ew_cov) | running means, variances, covariances, correlations, partial correlations, Mahalanobis distances and principal components | [entry](MODELS.md#ew_cov--exponentially-weighted-moments) · [flag joint outliers](how-to/flag-joint-outliers.md) · [track principal components](how-to/track-principal-components.md) · [find which series leads](how-to/find-which-series-leads.md) |
| [`rheon.spec.pca`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.pca) | principal components learned a row at a time, in state of the components times the columns: the incremental SVD, CCIPCA or Oja's rule | [entry](MODELS.md#pca--streaming-principal-components) · [track principal components](how-to/track-principal-components.md) |
| [`rheon.spec.ewm_moments`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ewm_moments) | exponentially weighted or windowed moments of columns to any order, each leg perhaps lagged: skew, kurtosis, cumulants, co-moments, autocovariances and autocorrelations, each a term you name | [entry](MODELS.md#ewm_moments--moments-to-any-order-lagged-or-not) · [OUTPUTS.md](OUTPUTS.md#ewm_moments) |
| [`rheon.spec.marginal`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.marginal) | every feature and target pair's moments, slope and t, kept in the state and read back as a table | [entry](MODELS.md#marginal--every-pairs-moments-kept-in-the-state) · [screen features against a target](how-to/screen-features-against-a-target.md) |
| [`rheon.spec.dependence`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.dependence) | nonlinear dependence between every feature and target pair: binned mutual information, Miller-Madow MI, Spearman's rho, Kendall's tau-b and Chatterjee's xi, per row and as a table | [entry](MODELS.md#dependence--nonlinear-dependence-mutual-information-and-rank-correlations) · [OUTPUTS.md](OUTPUTS.md#dependence) |
| [`rheon.spec.deco`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.deco) | one correlation for the whole matrix, or one per block and pair of blocks | [entry](MODELS.md#deco--one-correlation-for-the-whole-matrix) · [track the average correlation](how-to/track-the-average-correlation.md) |
| [`rheon.spec.rcov`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.rcov) | a block's realized covariance, robust to microstructure noise | [entry](MODELS.md#rcov--a-blocks-realized-covariance-robust-to-noise) · [a block's covariance from ticks](how-to/estimate-a-blocks-covariance-from-ticks.md) |
| [`rheon.spec.audit`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.audit) | a count of what each column holds, read by `ModelBank.check` | [entry](MODELS.md#audit--what-a-streams-columns-hold) · [check data quality](how-to/check-data-quality.md) |
| [`rheon.spec.kmeans`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.kmeans) | exponentially weighted k-means, each row labeled before it is learned | [entry](MODELS.md#kmeans--exponentially-weighted-k-means) · [keep centers on the data](how-to/keep-kmeans-centers-on-the-data.md) |
| [`rheon.spec.micro`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.micro) | density-based clustering into clusters of any shape and number | [entry](MODELS.md#micro--density-based-clustering-any-shape) · [choose `eps`](how-to/choose-micro-eps.md) |
| [`rheon.spec.ew_class`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ew_class) | Gaussian classification, one set of running moments per class | [entry](MODELS.md#ew_class--gaussian-classification-on-ew_cov-moments) · [choose a covariance type](how-to/choose-a-covariance-type.md) |
| [`rheon.spec.hoeffding_tree_classifier`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.hoeffding_tree_classifier) | an incremental decision tree for classification, splitting by information gain, Gini or Hellinger under a Hoeffding bound within a per-group byte cap | [entry](MODELS.md#hoeffding_tree_classifier--incremental-classification-tree) |
| [`rheon.spec.seqtest`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.seqtest) | a sequential test of a sign by betting, read at any row | [entry](MODELS.md#seqtest--a-sequential-test-of-a-sign-by-betting) · [read a sequential test](how-to/read-a-sequential-test.md) · [compare two models](how-to/compare-two-models.md) |
| [`rheon.spec.corrchange`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.corrchange) | tests of whether the correlations between the features have changed | [entry](MODELS.md#corrchange--has-the-correlation-structure-changed) · [detect a break](how-to/detect-a-break.md) |
| [`rheon.spec.bocpd`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.bocpd) | Bayesian online changepoint detection: how long the current regime has lasted, with constant, clock, or negative-binomial hazards and unknown- or known-variance emissions | [entry](MODELS.md#bocpd--how-long-has-this-regime-lasted) · [choose a changepoint hazard](how-to/choose-a-changepoint-hazard.md) |
| [`rheon.spec.hmm`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.hmm) | a Gaussian hidden Markov model, filtered and optionally fixed-lag smoothed as the stream runs, with full, tied, diag, or spherical covariances and partial learning | [entry](MODELS.md#hmm--which-regime-are-we-in) · [keep its states on their regimes](how-to/keep-hmm-states-on-their-regimes.md) |
| [`rheon.spec.hawkes`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.hawkes) | a self-exciting point process, fitted exactly by least squares | [entry](MODELS.md#hawkes--how-much-does-each-event-excite-the-next) |
| [`rheon.spec.spectral`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.spectral) | the power of each series at chosen periods, an exponentially weighted Fourier transform read as scipy's periodogram | [entry](MODELS.md#spectral--how-strong-is-each-cycle) |
| [`rheon.spec.garch`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.garch) | the GARCH family: a conditional variance for each return series, its parameters learned a row at a time | [entry](MODELS.md#garch--how-large-will-the-next-return-be) |
| [`rheon.spec.dcc`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.dcc) | dynamic conditional correlation: the full correlation matrix of a few return series, moving every row, its parameters given or learned a row at a time | [entry](MODELS.md#dcc--how-do-the-correlations-move-from-row-to-row) |
| [`rheon.spec.realized`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.realized) | realized volatility and the jump tests on it: realized variance, bipower and tripower variation, tripower quarticity, Barndorff-Nielsen and Shephard's test, Lee and Mykland's statistic and flag | [entry](MODELS.md#realized--how-much-did-it-move-and-did-it-jump) |
| [`rheon.spec.power_variation`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.power_variation) | power variation and kernel-weighted autocovariance sums of returns, named terms: multipower terms, kernel sums (flat-top or centered, the realized kernel and the HAC long-run variance), subsampled scales, and the count of returns | [entry](MODELS.md#power_variation--power-variation-and-kernel-weighted-autocovariance) |
| [`rheon.spec.tdigest`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.tdigest) | quantiles of each column at the levels you list, from a merging t-digest, uniform in rank | [entry](MODELS.md#tdigest--where-do-this-columns-quantiles-stand) |
| [`rheon.spec.ddsketch`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.ddsketch) | quantiles and tail means of each column within a relative accuracy, from a DDSketch, over every row or a window | [entry](MODELS.md#ddsketch--how-large-are-this-columns-quantiles-and-tails) |
| [`rheon.spec.bandit`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.bandit) | disjoint linear contextual bandit (LinUCB, linear Thompson sampling, epsilon-greedy) over logged arm and reward streams | [entry](MODELS.md#bandit--linear-contextual-bandits) · [OUTPUTS.md](OUTPUTS.md#bandit) |
| [`rheon.spec.half_space_trees`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.half_space_trees) | streaming Half-Space Trees unsupervised anomaly detector over bounded numeric features | [entry](MODELS.md#half_space_trees--streaming-half-space-trees-anomaly-detection) · [OUTPUTS.md](OUTPUTS.md#half_space_trees) |
| [`rheon.spec.random_cut_forest`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.random_cut_forest) | streaming Robust Random Cut Forest unsupervised anomaly detector with optional shingling | [entry](MODELS.md#random_cut_forest--robust-random-cut-forest-anomaly-detection) · [OUTPUTS.md](OUTPUTS.md#random_cut_forest) |

## Running a bank

A [model bank](GLOSSARY.md#the-model-bank-and-what-it-reads) fits a set of
specs together over the same rows. It runs inside a Polars query, in your
own loop, or from the command line ([The command line](#the-command-line)).

| feature | what it does | shown in |
|---|---|---|
| [`lf.rheon.fit_predict`](https://rheon-dev.github.io/rheon/namespaces.html#rheon.polars._frame.LazyFrameRheonNamespace.fit_predict) | adds each spec's output column to a query's rows, learning them as the query runs | [Fit a model bank over parquet files in a query](tutorials/stream-from-parquet.md) · [Run a bank in a query](how-to/run-a-bank-in-a-query.md) |
| [`lf.rheon.predict`](https://rheon-dev.github.io/rheon/namespaces.html#rheon.polars._frame.LazyFrameRheonNamespace.predict) | predicts a query's rows from a saved state or a bank, learning nothing | [Serve from a saved state](how-to/serve-from-a-saved-state.md) |
| [`lf.rheon.unnest`](https://rheon-dev.github.io/rheon/namespaces.html#rheon.polars._frame.LazyFrameRheonNamespace.unnest) | turns each spec's output column into one column per field | [Run a bank in a query](how-to/run-a-bank-in-a-query.md) |
| [`rp.fit_predict`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.fit_predict), [`rp.predict`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.predict), [`rp.unnest`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.unnest) | the same three as functions that take the frame first, which a type checker can see | [Run a bank in a query](how-to/run-a-bank-in-a-query.md#call-the-functions-a-type-checker-sees) |
| [`rp.ModelBank`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank) | a set of specs fitted together, fed one chunk at a time from your own code | [Run a bank in a Python loop](how-to/run-a-bank-in-a-python-loop.md) |
| [`ModelBank.fit_predict`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.fit_predict) | learns a frame's rows and returns them with each spec's output | [Run a bank in a Python loop](how-to/run-a-bank-in-a-python-loop.md) |
| [`ModelBank.fit_predict_batches`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.fit_predict_batches) | learns a query or a frame one chunk at a time, and yields each chunk's output | [Run a bank in a Python loop](how-to/run-a-bank-in-a-python-loop.md) |
| [`ModelBank.fit`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.fit) | learns every row and keeps only the state, for a run whose product is the state | [Run a bank in a Python loop](how-to/run-a-bank-in-a-python-loop.md) |
| [`ModelBank.predict`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.predict) | predicts a frame's rows from the bank as it stands, learning nothing | [Serve from a saved state](how-to/serve-from-a-saved-state.md) |
| [`ModelBank.fit_predict_arrow`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.fit_predict_arrow), [`ModelBank.predict_arrow`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.predict_arrow), [`rheon.ArrowStruct`](https://rheon-dev.github.io/rheon/rheon.html#rheon.ArrowStruct) | each spec's output as an Arrow struct array, read once by any consumer of the Arrow PyCapsule interface | [Hand the output to Arrow](how-to/hand-the-output-to-arrow.md) |
| [`rheon.thread_pool_size`](https://rheon-dev.github.io/rheon/rheon.html#rheon.thread_pool_size), [`ModelBank.threads`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.threads) | the number of threads a bank runs on, and the `threads` it was made with | [Use more cores](how-to/use-more-cores.md) · [`ModelBank.threads`' reference entry](https://rheon-dev.github.io/rheon/api/rheon.ModelBank.threads.html) |
| `chunk_size` | the rows a query feeds a bank at a time, which never changes the numbers | [Set the chunk size](how-to/set-the-chunk-size.md) |

## Saving, loading and serving

A bank's state, everything it has learned, can be saved to one file and
loaded back, on any operating system, to go on learning or to serve.

| feature | what it does | shown in |
|---|---|---|
| [`ModelBank.save`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.save), [`ModelBank.save_bytes`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.save_bytes) | writes the state, everything the bank has learned, to a file or to bytes; a query takes `save_state=` | [Save a state](how-to/save-a-state.md) |
| [`ModelBank.load`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.load), [`ModelBank.load_bytes`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.load_bytes) | reads a state back, on any operating system; a query takes `load_state=` | [Resume from a saved state](how-to/resume-from-a-saved-state.md) · [Save a state](how-to/save-a-state.md) |
| [`ModelBank.skip_learned`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.skip_learned) | keeps only the rows of an input the state has not learned, to resume on input that overlaps it | [Resume from a saved state](how-to/resume-from-a-saved-state.md) |
| [`ModelBank.write_json`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.write_json) | writes the state as JSON, for a program without this library | [Read a state without this library](how-to/read-a-state-without-this-library.md) |

## Reading a bank

A bank describes what it holds, live or loaded from its file, and names
the fields of its output.

| feature | what it does | shown in |
|---|---|---|
| [`ModelBank.coef`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.coef) | every fit's coefficients, one row per coefficient | [Read the coefficients](how-to/read-the-coefficients.md) |
| [`ModelBank.groups`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.groups), [`ModelBank.drop_groups`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.drop_groups) | the groups the bank holds a model for, and their removal | [Fit one model per group](how-to/fit-one-model-per-group.md) · [Run a bank in a Python loop](how-to/run-a-bank-in-a-python-loop.md) |
| [`ModelBank.summary`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.summary), [`ModelBank.describe`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.describe), [`ModelBank.last_row`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.last_row) | what each model was fed and how ready it is, each input column's statistics, and each model's last output row | [Inspect a bank](how-to/inspect-a-bank.md) |
| [`ModelBank.specs`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.specs), [`ModelBank.rows_fed`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.rows_fed), [`ModelBank.output_fields`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.output_fields) | the specs a bank runs with every field filled in, the rows fed so far, and each spec's output fields | [Inspect a bank](how-to/inspect-a-bank.md) |
| [`ModelBank.solve_failures`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.solve_failures) | how many of each model's solves needed jitter, kept the previous fit, or stopped at a pass cap | [Inspect a bank](how-to/inspect-a-bank.md) |
| [`ModelBank.closed_groups`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.closed_groups) | one row per closed group, with its running sums | [Close groups and read each one's row](how-to/close-groups.md) |
| [`ModelBank.gram`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.gram) | the Gram behind a spec's fit, per group, as numpy arrays | [Keep a Gram per block in one pass](how-to/fit-weekly-grams-in-one-pass.md) · [Store a compact Gram](how-to/store-a-compact-gram.md) |
| [`ModelBank.marginal`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.marginal) | the feature and target pairs a `marginal` spec keeps, one row per pair | [Screen features against a target](how-to/screen-features-against-a-target.md) |
| [`ModelBank.dependence`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.dependence) | the feature and target pairs a `dependence` spec keeps, one row per pair, with fresh exact rank reads | [entry](MODELS.md#dependence--nonlinear-dependence-mutual-information-and-rank-correlations) |
| [`ModelBank.vector_ar`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.vector_ar) | a `vector_ar` spec's impulse responses and their standard errors, its variance decomposition, roots, long-run mean, every order's criteria and a Granger test asked after the fact, read from its fit as it stands | [entry](MODELS.md#vector_ar--a-vector-autoregression-with-granger-tests) |
| [`ModelBank.smoothed_tail`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.smoothed_tail) | a `kalman` spec's last observations under `smooth_lag`, smoothed with every row so far, read from the state | [The fixed-lag smoother](MODELS.md#the-fixed-lag-smoother) |
| [`ModelBank.audit`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.audit), [`ModelBank.check`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ModelBank.check) | what an `audit` spec counted, and the findings about the data each model was fed, the worst first | [Check a stream's data quality](how-to/check-data-quality.md) · [DATA-QUALITY.md](DATA-QUALITY.md) |
| [`rheon.spec.output_fields`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.output_fields), [`rheon.spec.output_index`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.output_index) | the names of the fields a spec writes, and the values each name encodes | [Find an output field by what it holds](how-to/find-an-output-field.md) · [OUTPUTS.md](OUTPUTS.md) |
| [`rheon.spec.coef_fields`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.coef_fields), [`rheon.spec.coef_index`](https://rheon-dev.github.io/rheon/spec.html#rheon.spec.coef_index) | every coefficient a spec reports, with its place in `coef` and its unnested column | [Find an output field by what it holds](how-to/find-an-output-field.md) · [Read the coefficients](how-to/read-the-coefficients.md) |

## Preparing a stream

The window operators return Polars expressions, computed over a window of
the clock and run in `lf.rheon.with_windows`. `ewm_mean`, `ewm_sum`,
`ewm_rate` and `ewm_quantile` look back from a row, or ahead under
`reverse=True`, and a look-ahead one can be a spec's target. `ewm_var`,
`ewm_std` and `ewm_moments` look back only ([WINDOWS.md](WINDOWS.md) describes
each).

| feature | what it does | shown in |
|---|---|---|
| [`rp.ops.ewm_mean`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.ewm_mean) | the time-weighted mean (or point-mass EW mean under `adjust=True`) of a column over each row's window | [Make features that look back](how-to/make-window-features.md) · [Make a target that looks ahead](how-to/make-a-target-that-looks-ahead.md) |
| [`rp.ops.ewm_sum`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.ewm_sum) | the weighted sum over each row's window, such as the two sums of a VWAP | [Make features that look back](how-to/make-window-features.md#make-a-weighted-mean-such-as-a-vwap) |
| [`rp.ops.ewm_rate`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.ewm_rate) | a weighted sum per unit of clock time, such as volume per second | [WINDOWS.md](WINDOWS.md) |
| [`rp.ops.ewm_var`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.ewm_var), [`rp.ops.ewm_std`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.ewm_std) | the variance and standard deviation about the time-weighted mean | [WINDOWS.md](WINDOWS.md) · [Scale a feature by its spread](how-to/scale-a-feature-by-its-spread.md) |
| [`rp.ops.ewm_moments`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.ewm_moments) | one term of the moments of one or more columns over each row's window, each weighing its held interval: a skew, a kurtosis, a cumulant, a co-moment, an autocorrelation | [WINDOWS.md](WINDOWS.md) · [the model's entry](MODELS.md#ewm_moments--moments-to-any-order-lagged-or-not) |
| [`rp.ops.ewm_quantile`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.ewm_quantile) | a weighted quantile of the window's values, such as a median | [Weight rows by a robust scale](how-to/weight-rows-by-a-robust-scale.md) |
| [`rp.ops.rolling_quantile`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.rolling_quantile), [`rp.ops.rolling_rank`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.rolling_rank) | exact rolling quantile (all six Polars interpolation rules) and exact rolling rank (`"average"`, `"min"`, `"max"`, `"dense"`) over a window | [WINDOWS.md](WINDOWS.md) |
| [`rp.ops.ewm_profile`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.ewm_profile) | an exponentially weighted mean, variance or standard deviation by each row's phase of a period, such as the hour of the day on a zoned clock's wall clock | [`ewm_profile`'s reference entry](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.ewm_profile) |
| [`rp.ops.rolling_max`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.rolling_max); [`rp.ops.rolling_min`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.rolling_min) | the largest or smallest value of a window, or since an event, such as a drawdown's high-water mark | [The largest and smallest value](WINDOWS.md#the-largest-and-smallest-value-and-since-an-event) |
| [`rp.ops.fourier`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.fourier) | the sine and cosine of each row's phase in a fixed period, such as the hour of the day, as features of a regression; no state, and statsmodels' `Fourier` terms to the last bit | [`fourier`'s reference entry](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.fourier) |
| [`rp.ops.calendar_fourier`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.calendar_fourier) | the same for a calendar hour, day, week, month, quarter or year; statsmodels' `CalendarFourier` to the last bit | [`calendar_fourier`'s reference entry](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.calendar_fourier) |
| [`rp.ops.stl`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.stl), [`rp.ops.mstl`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.mstl) | the trend, a season or the remainder of STL or MSTL on each group's last rows, read a few rows late, where the decomposition has stopped moving; statsmodels' `STL` and `MSTL`, within the rounding of their own operations | [`stl`'s reference entry](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.stl) |
| [`rp.ops.increment`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.increment) | a column's step from its last value within the group and session | [WINDOWS.md](WINDOWS.md) |
| [`rp.ops.invert`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.invert), [`rp.ops.unknown`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.unknown) | the root of an equation of the row in one unknown (`rp.ops.unknown()`), `f(x) = equals` on a bracket, solved row by row in the run that computes the window operators: given only where the function's own rounding fixes it to the tolerance, and null with a reason otherwise, such as the volatility a pricing formula gives a quoted price | [`invert`'s reference entry](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.invert) |
| [`rp.ops.cross_mean`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.cross_mean), [`rp.ops.cross_std`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.cross_std), [`rp.ops.cross_rank`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.cross_rank), [`rp.ops.cross_quantile`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.cross_quantile), [`rp.ops.cross_ref`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.cross_ref) | a column's mean, standard deviation and quantile across every group's row at each stamp of the clock, or across the rows of the groups each row lists as its neighbors, broadcast back to each row; each row's rank within it; and one reference row's value read by every row of its stamp | [WINDOWS.md](WINDOWS.md) |
| [`rp.ops.tick_sign`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.tick_sign) | the tick rule: a trade's sign, +1 or -1, from its price's last move, the last nonzero sign held over a step of 0 | [WINDOWS.md](WINDOWS.md) |
| [`rp.ops.run_length`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.run_length) | the rows a value has held, ending at each row, to find a frozen feed | [WINDOWS.md](WINDOWS.md#a-streams-health-run-lengths-and-two-clocks) |
| [`rp.ops.clock_lag`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.clock_lag), [`rp.ops.clock_drift`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.clock_drift) | the decayed mean lag of each row's arrival behind its event, and how fast it grows | [WINDOWS.md](WINDOWS.md#a-streams-health-run-lengths-and-two-clocks) |
| [`rp.ops.clock_step`](https://rheon-dev.github.io/rheon/ops.html#rheon.polars.ops.clock_step) | the decayed mean, variance or standard deviation of the steps between each time and the one before it, or the decayed count of its repeats or of its steps back | [WINDOWS.md](WINDOWS.md#a-streams-health-run-lengths-and-two-clocks) |
| [`lf.rheon.with_windows`](https://rheon-dev.github.io/rheon/namespaces.html#rheon.polars._frame.LazyFrameRheonNamespace.with_windows), [`rp.stream.with_windows`](https://rheon-dev.github.io/rheon/stream.html#rheon.polars.stream.with_windows) | adds window columns to a stream in one pass, before the bank | [Make features that look back](how-to/make-window-features.md) · [Resume a window run](how-to/resume-a-window-run.md) |
| [`rp.stream.refresh_time`](https://rheon-dev.github.io/rheon/stream.html#rheon.polars.stream.refresh_time) | puts series that tick at their own times on one refresh grid, with each value's age on the clock under `age=True` | [Align series that tick at their own times](how-to/align-series-that-tick-at-their-own-times.md) |
| [`rp.stream.merge`](https://rheon-dev.github.io/rheon/stream.html#rheon.polars.stream.merge) | merges streams that each run in clock order into one, every row kept and tagged with its stream; `carry` fills columns forward, each with its age on the clock | [Merge streams by the clock](how-to/merge-streams-by-the-clock.md) |
| [`rp.stream.bars`](https://rheon-dev.github.io/rheon/stream.html#rheon.polars.stream.bars) | aggregates a stream into bars that close when a running count, sum or signed sum reaches a threshold: tick, volume, dollar and imbalance bars | [WINDOWS.md](WINDOWS.md) |
| [`rp.stream.save_checkpoint`](https://rheon-dev.github.io/rheon/stream.html#rheon.polars.stream.save_checkpoint), [`rp.stream.load_checkpoint`](https://rheon-dev.github.io/rheon/stream.html#rheon.polars.stream.load_checkpoint) | saves the states of a run's stages, such as a grid, a window run and a bank, as one file written in full or not at all, and reads each stage's state back | [Saved states](STATE-WORKFLOW.md#a-pipeline-checkpoint) |
| [`rp.stream.embargo`](https://rheon-dev.github.io/rheon/stream.html#rheon.polars.stream.embargo), [`rp.stream.ROLE`](https://rheon-dev.github.io/rheon/stream.html#rheon.polars.stream.ROLE) | doubles a stream so each row's target is learned a delay after its row, marking each copy in a column named by `ROLE` | [Learn targets known late](how-to/learn-targets-known-late.md) |
| [`rheon.target`](https://rheon-dev.github.io/rheon/rheon.html#rheon.target) | a target column whose output fields take a name of your own | [Evaluate an output frame](how-to/evaluate-an-output-frame.md) |
| [`rheon.Target`](https://rheon-dev.github.io/rheon/rheon.html#rheon.Target), [`rheon.FormulaTarget`](https://rheon-dev.github.io/rheon/rheon.html#rheon.FormulaTarget) | the types of a named target and of a formula target written as a table, as a spec dict and the command line's TOML hold them | [Specs in TOML](RUNNER.md#specs-in-toml), which writes a formula target |

## Scoring and comparing

`rp.eval` reads a bank's output, in memory or as sums kept chunk by chunk.

| feature | what it does | shown in |
|---|---|---|
| [`rp.eval.metrics`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.metrics), [`rp.eval.window_metrics`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.window_metrics), [`rp.eval.compare_specs`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.compare_specs) | a spec's out-of-sample scores over a frame or a query of one (including opt-in `mae`, pinball loss and D² pinball), in windows of the clock, and several specs' in one table | [Evaluate an output frame](how-to/evaluate-an-output-frame.md) |
| [`rp.eval.interval_metrics`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.interval_metrics) | post-run coverage, mean width, Winkler score and size-stratified coverage (SSC) for conformal intervals, single- or multi-level | [Is another model better?](DIAGNOSTICS.md#is-another-model-better) |
| [`rp.eval.sums`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.sums), [`rp.eval.merge_sums`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.merge_sums), [`rp.eval.from_sums`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.from_sums), [`rp.eval.SUM_FIELDS`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.SUM_FIELDS) | scores from sums that add up across chunks and runs, over a frame or a query, and the names of those sums | [Evaluate an output frame](how-to/evaluate-an-output-frame.md#score-an-output-too-large-to-hold) · [`SUM_FIELDS`' reference entry](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.SUM_FIELDS) |
| [`rp.eval.unpack`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.unpack), [`rp.eval.RESERVED`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.RESERVED) | a spec's output in long form, one row per input row and slot, and the column names it takes for itself | [Evaluate an output frame](how-to/evaluate-an-output-frame.md) · [`RESERVED`'s reference entry](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.RESERVED) |
| [`rp.eval.grid_search`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.grid_search) | scores every point of a grid in one pass, and chooses from the rows before each test span | [CALIBRATION.md](CALIBRATION.md) |
| [`rp.eval.diebold_mariano`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.diebold_mariano), [`rp.eval.clark_west`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.clark_west), [`rp.eval.giacomini_white`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.giacomini_white), [`rp.eval.seqtest`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.seqtest), [`rp.eval.mean_seqtest`](https://rheon-dev.github.io/rheon/eval.html#rheon.polars.eval.mean_seqtest) | tests of equal unconditional or conditional predictive accuracy, anytime-valid loss-differential tests, and a sequential test of a sign, over a frame or a query of one | [Compare two models](how-to/compare-two-models.md) |

## Fits from Grams

A Gram is the running sums a least-squares fit is solved from. `rheon.gram`
fits from the Grams a bank keeps, without reading the rows again
([GRAM.md](GRAM.md)).

| feature | what it does | shown in |
|---|---|---|
| [`rheon.gram.Gram`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.Gram), [`rheon.gram.from_row`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.from_row), [`rheon.gram.INTERCEPT`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.INTERCEPT) | the mapping a bank's Gram is returned as, a closed group's row read as one, and the intercept column's name | [Keep a Gram per block in one pass](how-to/fit-weekly-grams-in-one-pass.md) · [Close groups](how-to/close-groups.md) · [`INTERCEPT`'s reference entry](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.INTERCEPT) |
| [`rheon.gram.merge`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.merge), [`rheon.gram.subset`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.subset) | pools the Grams of several blocks, and keeps some of a Gram's columns | [Pool Grams](how-to/pool-grams.md) |
| [`rheon.gram.solve`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.solve), [`rheon.gram.solve_subsets`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.solve_subsets), [`rheon.gram.coef_stats`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.coef_stats) | ridge coefficients from a Gram, for many subsets of blocks at once, and their standard errors and t | [Fit a ridge from a Gram](how-to/fit-a-ridge-from-a-gram.md) · [Select columns stably over weeks](how-to/select-columns-stably-over-weeks.md) |
| [`rheon.gram.lasso_path`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.lasso_path), [`rheon.gram.lars_path`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.lars_path), [`rheon.gram.lars_paths`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.lars_paths) | the elastic-net path by coordinate descent, and the lasso path by least angle regression | [Choose columns on the lasso path](how-to/choose-columns-on-the-lasso-path.md#choose-columns-on-a-grams-lasso-path) |
| [`rheon.gram.condition`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.condition), [`rheon.gram.vif`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.vif), [`rheon.gram.correlation`](https://rheon-dev.github.io/rheon/gram.html#rheon.gram.correlation) | collinearity diagnostics, variance inflation factors and the correlation matrix of a Gram's columns | [Check a fit is determined](how-to/check-a-fit-is-determined.md) |

## Correlation matrices

`rheon.corr` reads a correlation matrix from an array, a Gram or a closed
group's row.

| feature | what it does | shown in |
|---|---|---|
| [`rheon.corr.nearest`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.nearest), [`rheon.corr.shrink`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.shrink) | the nearest correlation matrix, and shrinkage toward a structured target | [Repair and score a correlation matrix](how-to/repair-and-score-a-correlation-matrix.md) |
| [`rheon.corr.loss`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.loss), [`rheon.corr.fisher_se`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.fisher_se), [`rheon.corr.to_z`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.to_z), [`rheon.corr.from_z`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.from_z), [`rheon.corr.Z_CLIP`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.Z_CLIP) | a predicted matrix scored against the realized one, a correlation's standard error, and Fisher's transform and its inverse, clipped at `Z_CLIP` | [Repair and score a correlation matrix](how-to/repair-and-score-a-correlation-matrix.md) |
| [`rheon.corr.spectral`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.spectral), [`rheon.corr.from_spectral`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.from_spectral), [`rheon.corr.mp_edge`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.mp_edge), [`rheon.corr.mp_density`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.mp_density) | the top eigenpairs, a matrix rebuilt from them, and the Marchenko-Pastur noise edges and density | [Repair and score a correlation matrix](how-to/repair-and-score-a-correlation-matrix.md#choose-another-function) |
| [`rheon.corr.absorption`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.absorption), [`rheon.corr.absorption_shift`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.absorption_shift) | the absorption ratio, and its standardized shift | [Repair and score a correlation matrix](how-to/repair-and-score-a-correlation-matrix.md#choose-another-function) |
| [`rheon.corr.equicorr`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.equicorr), [`rheon.corr.equicorr_row`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.equicorr_row), [`rheon.corr.equicorr_loglik`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.equicorr_loglik), [`rheon.corr.block_means`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.block_means), [`rheon.corr.from_blocks`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.from_blocks), [`rheon.corr.signal_share`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.signal_share) | one correlation for a whole matrix or per block, and how much of its movement across blocks is not noise | [Track the average correlation](how-to/track-the-average-correlation.md) · [Choose another function](how-to/repair-and-score-a-correlation-matrix.md#choose-another-function) |
| [`rheon.corr.epps_invert`](https://rheon-dev.github.io/rheon/corr.html#rheon.corr.epps_invert) | the correlation over several rows from lagged co-moments, undoing the Epps effect | [REGIMES.md](REGIMES.md), [Series that tick at their own times](explanation/asynchronous-series.md) |

## Simulation

One module sits beside the bank: `rp.sim` simulates streams whose truth is
known. (The helper that builds training data for gradient-boosted trees is
internal, and has no public name.)

| feature | what it does | shown in |
|---|---|---|
| [`rp.sim.regimes`](https://rheon-dev.github.io/rheon/sim.html#rheon.polars.sim.regimes) | a simulated stream whose correlation changes by regime, with its truth | [Test a detector on known truth](how-to/test-a-detector-on-known-truth.md) · [REGIMES.md](REGIMES.md) |

## Parameters every spec takes

Every model takes these, apart from those its docstring says it refuses
([What every spec takes](MODELS.md#what-every-spec-takes)). A model's own
parameters are on its builder's reference page.

| family | parameters | what they set | shown in |
|---|---|---|---|
| the columns | `targets`, `features`, `fit_intercept` | what the model reads, and whether the fit has a level of its own | [What a spec names](MODELS.md#what-a-spec-names) |
| the clock and its decay | `clock`, `half_life`, `lam`, `gap_cap` | how far apart two rows are, how fast the past fades, and the longest step a gap counts as | [The clock and its units](MODELS.md#the-clock-and-its-units) · [Time and decay](explanation/time-and-decay.md) |
| a hard window, in `ewridge`, `lasso`, `ar`, `vector_ar`, `ew_cov`, `ew_class`, `marginal` and `spectral`, cut from snapshots; `expectile`, `isotonic`, `krls`, `dependence`, `hawkes`, `realized`, `power_variation`, `ddsketch`, `ewm_moments` and `corrchange` take a `window_size` of their own | `window_size`, `closed`, `snapshot_every`, `max_rows_between_snapshots`, `window_budget` | the span past which a row counts for nothing | [A hard window](MODELS.md#a-hard-window) · [Forget rows past a window](how-to/forget-rows-past-a-window.md) |
| the clock's policy | `session`, `session_gap`, `restart_after_step_back` | what a session boundary and a step back do | [Handle sessions, gaps and steps back](how-to/handle-sessions-gaps-and-steps-back.md) |
| groups | `group`, `group_close` | one model per value of a column, and when a group closes | [Fit one model per group](how-to/fit-one-model-per-group.md) · [Close groups](how-to/close-groups.md) |
| weights | `weight` | how much each row counts | [Weight rows](how-to/weight-rows.md) |
| warm-up | `min_weight`, `min_settled_frac`, `max_error_inflation`, `emit_error_inflation`, `emit_target_weight` | when a model has seen enough to report, and each target's weight its floor reads | [Warm-up gates](MODELS.md#warm-up-gates) · [Decide when a model is ready](how-to/decide-when-a-model-is-ready.md) |
| targets known late | `embargo`, `label_clock`, `max_held_rows`, `held_overflow` | when each row's target is learned, and how many rows may wait | [Learn targets known late](how-to/learn-targets-known-late.md) |
| coefficients | `coef_every`, `max_rows_between_coefs`, `emit_se_coef` | which rows carry the coefficients, and their standard errors | [Read the coefficients](how-to/read-the-coefficients.md) · [Check standard errors](how-to/check-standard-errors.md) |
| clocks in the output | `emit_clocks` | each row's clock, and the clock it was learned at | [Learn targets known late](how-to/learn-targets-known-late.md#give-every-row-the-same-delay) |
| the residuals | `emit_sigma`, `emit_zscore`, `emit_metrics`, `metric_quantiles`, `emit_autocorr`, `resid_autocorr_lag`, `resid_quantiles` | the residuals' spread, each one's z-score, running accuracy and loss scores, autocorrelation and quantiles | [DIAGNOSTICS.md](DIAGNOSTICS.md) |
| intervals and p-values | `conformal`, `conformal_symmetric`, `conformal_rate`, `emit_conformal_p`, `conformal_p_window_rows`, `conformal_p_window_size` | symmetric or asymmetric conformal intervals at one or more levels, and standalone out-of-sample conformal p-values | [Put an interval on a prediction](how-to/put-an-interval-on-a-prediction.md) · [DIAGNOSTICS.md](DIAGNOSTICS.md#conformal-intervals) |
| state budget | `max_state_bytes` | per-group memory cap for windowed diagnostic and model state | [DIAGNOSTICS.md](DIAGNOSTICS.md#conformal-intervals) |
| a grid's choice | `emit_selected`, `emit_averaged`, `average_eta` | the grid point that has predicted best so far, or the grid's predictions averaged, as the stream runs | [Choose among a grid's settings](how-to/choose-among-a-grid-online.md) |
| drift | `emit_drift`, `drift_delta`, `drift_threshold`, `drift_action`, `drift_kind`, `drift_options` | a test of the residuals' level, and whether a flag resets the model; or Page's CUSUM, ADWIN, DDM, EDDM or KSWIN in its place | [Detect a break](how-to/detect-a-break.md) · [Measure heavy tails](how-to/measure-heavy-tails.md) |
| calibration | `emit_calibration`, `emit_spiegelhalter`, `calibration_half_life` | whether the predictions are the right size, and Spiegelhalter's z-test on binary probabilities | [Check the scale](how-to/check-the-scale.md) |
| breaks | `emit_breaks`, `breaks_half_life` | tests of whether the relationship has broken | [Detect a break](how-to/detect-a-break.md) · [Has it broken?](DIAGNOSTICS.md#has-it-broken) |
| robust standard errors | `emit_robust_se`, `robust_se_half_life`, `robust_se_lags`, `robust_se_types`, `max_state_bytes` | standard errors that allow for uneven and correlated residuals, with HC0–HC3 leverage variants and a group state cap | [Check standard errors](how-to/check-standard-errors.md) · [Can its t be trusted?](DIAGNOSTICS.md#can-its-t-be-trusted) |
| specification | `emit_specification`, `emit_arch_lm`, `arch_lm_lags`, `emit_white`, `white_interaction_terms`, `specification_half_life`, `ljung_box_lags` | tests of what a linear model leaves out: Ljung-Box, Breusch-Pagan, RESET, Engle's ARCH LM and White's test | [Find what a model misses](how-to/find-what-the-model-misses.md) · [What is it missing?](DIAGNOSTICS.md#what-is-it-missing) |
| a target's horizon | `horizon_rows` | the rows a target looks ahead, which the diagnostics allow for | [Can its t be trusted?](DIAGNOSTICS.md#can-its-t-be-trusted) |
| tails | `emit_tails`, `tails_half_life` | how heavy the residuals' tails are | [Measure heavy tails](how-to/measure-heavy-tails.md) · [Are its tails heavy?](DIAGNOSTICS.md#are-its-tails-heavy) |
| influence | `emit_influence`, `emit_cooks_distance`, `emit_dfbetas`, `influence_half_life` | how far each row moved the fit (DFFITS, Cook's distance and per-coefficient DFBETAS) | [Find an influential row](how-to/find-an-influential-row.md) · [Which row moved it?](DIAGNOSTICS.md#which-row-moved-it) |
| feature health | `emit_feature_health`, `feature_health_half_life` | whether a feature has frozen or moved | [Check feature health](how-to/check-feature-health.md) |

## Errors and warnings

Each error and warning says where it happened, what is wrong and what to
do. The library's errors subclass Python's and Polars' own, and its
warnings Python's, so a handler you already have catches them.

| feature | what it is | shown in |
|---|---|---|
| [`rheon.ColumnNotFoundError`](https://rheon-dev.github.io/rheon/rheon.html#rheon.ColumnNotFoundError) | a column a call reads is not in its input | [What the bank refuses](MODELS.md#what-the-bank-refuses) |
| [`rheon.OutputColumnNotFoundError`](https://rheon-dev.github.io/rheon/rheon.html#rheon.OutputColumnNotFoundError) | a spec's output column is not in the frame a scoring function of `rp.eval` reads; a `ColumnNotFoundError` | [its reference entry](https://rheon-dev.github.io/rheon/api/rheon.OutputColumnNotFoundError.html) · [Evaluate an output frame](how-to/evaluate-an-output-frame.md) |
| [`rheon.WrongTypeError`](https://rheon-dev.github.io/rheon/rheon.html#rheon.WrongTypeError), [`rheon.RefusedValueError`](https://rheon-dev.github.io/rheon/rheon.html#rheon.RefusedValueError) | a value of the wrong type, a `TypeError`, and a refused value, a `ValueError` | [What the bank refuses](MODELS.md#what-the-bank-refuses) |
| [`rheon.RheonWarning`](https://rheon-dev.github.io/rheon/rheon.html#rheon.RheonWarning) | the base class of every warning the library issues, so `warnings.filterwarnings("ignore", category=rheon.RheonWarning)` silences them all | [its reference entry](https://rheon-dev.github.io/rheon/api/rheon.RheonWarning.html) |
| [`rheon.ReadinessWarning`](https://rheon-dev.github.io/rheon/rheon.html#rheon.ReadinessWarning) | a model is not ready, or may never become ready | [Decide when a model is ready](how-to/decide-when-a-model-is-ready.md) |
| [`rp.OrderNotGuaranteedWarning`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.OrderNotGuaranteedWarning) | Polars may change the row order of the query a bank reads | [Put rows in clock order](how-to/put-rows-in-clock-order.md) |
| [`rp.ConsumedSourceWarning`](https://rheon-dev.github.io/rheon/rheon.html#rheon.polars.ConsumedSourceWarning) | a query whose source is written in Python may give its rows only once | [Feed a bank from a database](how-to/feed-a-bank-from-a-database.md) |
| [`rheon.UnstableWarning`](https://rheon-dev.github.io/rheon/rheon.html#rheon.UnstableWarning), [`rheon.RheonDeprecationWarning`](https://rheon-dev.github.io/rheon/rheon.html#rheon.RheonDeprecationWarning) | a part of the API not yet promised to stay the same was used, or a name still read by a spelling it will stop reading | [VERSIONS.md](VERSIONS.md) |

## Versions and the environment

The library reports its versions, and reads four environment variables.

| feature | what it is | shown in |
|---|---|---|
| `rheon.__version__`, [`rheon.native_version`](https://rheon-dev.github.io/rheon/rheon.html#rheon.native_version) | the package's version, and its compiled extension's, which match in a consistent install | [`native_version`'s reference entry](https://rheon-dev.github.io/rheon/api/rheon.native_version.html) · [VERSIONS.md](VERSIONS.md), what a version promises |
| [`rheon.schema_version`](https://rheon-dev.github.io/rheon/rheon.html#rheon.schema_version) | the version of the state file's layout this build writes | [Resume from a saved state](how-to/resume-from-a-saved-state.md) · [A bank's state and resuming it](explanation/state-and-resuming.md) |
| `RHEON_MAX_THREADS` | the number of threads a bank runs on when it is given no `threads` | [Use more cores](how-to/use-more-cores.md) |
| `RHEON_TIMING` | prints where a run's time went, stage by stage | [PERFORMANCE.md](PERFORMANCE.md) |
| `RHEON_SKIP_CPU_CHECK` | skips the import-time check that an x86-64 CPU has the features the wheel was built for | [What the x86-64 wheels need](VERSIONS.md#what-the-x86-64-wheels-need) |
| `RHEON_WARN_UNSTABLE` | turns on an `UnstableWarning` for each use of an unstable part of the API | [VERSIONS.md](VERSIONS.md) |

## The command line

`rheon` runs a bank from a file to a file, with no Python, from a
TOML configuration file. Each `[[specs]]` table takes a spec's `name` and
the keywords every spec shares, and a `[specs.model]` table under it names
the model with `type` and takes the model's own keywords ([Run a bank from the command
line](how-to/run-a-bank-from-the-command-line.md), and every key and flag
in [RUNNER.md](RUNNER.md)).

| feature | what it does | shown in |
|---|---|---|
| `--config`, and the keys `input`, `output`, `specs` | the configuration file, the files it reads and writes, and its specs | [Run a bank from the command line](how-to/run-a-bank-from-the-command-line.md) |
| `--input`, `--output`, `--no-output`, `--input-format`, `--output-format`, `--output-layout`, and the keys `input_format`, `output_format`, `output_layout`, `keep_columns` | the files read and written, their formats, flat columns or one per spec, and which input columns pass through | [Flags and the TOML keys they override](RUNNER.md#flags-and-the-toml-keys-they-override) · [Input and output files](RUNNER.md#input-and-output-files) |
| `--chunk-size`, `--threads`, and the keys `chunk_size`, `threads` | the rows read at a time, and the bank's threads | [Memory, threads and chunk size](RUNNER.md#memory-threads-and-chunk-size) |
| `--load-state`, `--save-state`, `--skip-learned`, `--clock-runs-on`, `--predict`, and the keys `load_state`, `save_state`, `skip_learned`, `clock_runs_on`, `predict` | resume from a state, save one, skip the rows a state has learned, and serve without learning | [Resume from a saved state](how-to/resume-from-a-saved-state.md) · [Serve from a saved state](how-to/serve-from-a-saved-state.md) |
| `--closed-groups`, and the key `closed_groups` | writes each closed group's row to a file of its own as the run goes | [Closed groups to a sidecar file](RUNNER.md#closed-groups-to-a-sidecar-file) |
| `--dry-run`, `--quiet` (`-q`) | checks the configuration and the input without fitting, and turns the progress line off | [Run a bank from the command line](how-to/run-a-bank-from-the-command-line.md) |
| `--help` (`-h`), `--version` (`-V`) | the flags, and the program's version | [RUNNER.md](RUNNER.md) |
