"""The changepoint detectors' builders, :func:`bocpd` and :func:`corrchange`:
a part of :mod:`rheon._spec` moved to a file of its own, keeping
that one under ``tests/test_repo_hygiene.py``'s 250 KB cap for a source
file. :mod:`rheon.spec` documents them with the rest.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._duration import Duration
from rheon._kwargs import CommonKwargs
from rheon._spec import _checked, _common, _mirror_target


@_checked
def bocpd(
    name: str,
    *,
    features: list[str],
    hazard_rows: float | None = None,
    hazard: float | Duration | None = None,
    hazard_col: str | None = None,
    hazard_kind: Literal["constant", "neg_binomial"] = "constant",
    hazard_k: int | None = None,
    hazard_p: float | None = None,
    covariance_type: Literal["full", "diag"] = "diag",
    mean_prior: list[float] | None = None,
    mean_precision_prior: float | None = None,
    degrees_of_freedom_prior: float | None = None,
    covariance_prior: list[float] | None = None,
    obs_var: float | list[float] | None = None,
    robust_beta: float | None = None,
    prune_below: float | None = None,
    max_run: int | None = None,
    max_state_bytes: int | None = None,
    warm_rows: int | None = None,
    warm_span: float | Duration | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Bayesian online changepoint detection, a posterior over how long the current
    regime has lasted.

    Use it where the age of the current regime matters: it keeps a
    distribution over the run length, the rows since the last break, so "we
    are 40 rows into a regime" comes with every row, where a test answers
    only "has something changed?". It is Adams and MacKay's (2007)
    algorithm, over the features, with no target. Use :func:`corrchange`
    for a test of a change in correlation, and a regression's
    ``emit_breaks`` for a break in its coefficients.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`rheon.spec`. This model takes ``features`` and no
    ``targets``. ``half_life`` and ``lam`` are refused, since the run-length
    posterior is what forgets, and the hazard is how fast.
    ``fit_intercept`` has nothing to act on, as there is no regression.
    ``min_weight`` withholds what is reported, never what is learned. No
    diagnostic that reads residuals applies. This model's own:

    ``hazard_rows``, ``hazard``
        How likely a break is, per row or per unit of clock time:

        .. list-table::
           :header-rows: 1
           :widths: 18 40 42

           * - parameter
             - what it is
             - the chance of a break
           * - ``hazard_rows``, 250 by default
             - the expected rows between breaks, on any clock
             - ``H = 1 / hazard_rows`` after every row
           * - ``hazard``
             - ``tau``, the expected clock time between breaks: a number of
               the clock's units, or of rows where the spec has no clock, or
               a duration such as ``"1h"`` on a temporal clock
             - ``h = 1 - exp(-dt / tau)`` for the step of ``dt`` into each
               row, before the row is read

        Choose ``hazard_rows`` where a break is as likely after any row,
        however far apart the rows fall: Adams and MacKay's hazard, one step
        an observation. Choose ``hazard`` where breaks come in time, so a
        long gap makes a break near certain and a row sharing the last
        row's stamp cannot begin a run. On regular steps the two are the
        same model at ``hazard_rows = 1 / (1 - exp(-dt / tau))``: every
        output agrees but ``p_change``, which the per-row form reads with
        ``H`` of the mass at ``r = 0`` after the row, ``p_rows = H + (1 -
        H) * p_clock`` (*Reading p_change and run_mode* in Notes). On data
        with irregular steps, each form fits best where breaks come as it
        assumes, so the choice is a model of where breaks come from.

        ``hazard_rows`` is finite and above 1. ``hazard`` is a clock
        parameter like ``half_life``: a plain number is refused on a
        temporal clock, a duration without one, and either beside a clock
        parameter written the other way. It is finite and above 0. ``dt`` is
        the step decay reads, so a gap past ``gap_cap`` counts as the cap
        and a row of weight 0 is in no step. The two are refused together.
        *A clock hazard* in Notes gives the rules at a step of 0.
    ``hazard_col``
        Read a per-row hazard from this column instead, which the bank reads beside
        the features as it reads a weight column. The value at a row is the expected
        rows between breaks as of that row: ``1 / value`` is the chance of a break
        after the row, between it and the next. A null, NaN, ±inf or magnitude above
        ``1e100`` falls back to ``hazard_rows``, and any other value of 1 or
        less is refused by every call that reads the column,
        :meth:`rheon.polars.ModelBank.predict` included, naming the row.
        Default none. ``hazard`` puts its chance before each row, so the two
        are refused together.
    ``covariance_type``
        The covariance each run's predictive keeps, in scikit-learn's words,
        both exact conjugate updates:

        .. list-table::
           :header-rows: 1
           :widths: 18 40 42

           * - ``covariance_type``
             - the predictive
             - what it adds
           * - ``"diag"`` (the default)
             - a normal-inverse-gamma per feature
             -
           * - ``"full"``
             - a normal-inverse-Wishart over all of them
             - the one that can see a break in the correlation with the
               marginals unchanged: in ``run_mode``, since ``p_change`` never
               moves, and late (*Reading p_change and run_mode* in Notes)

        While the prior is set from the first rows (the default), both are
        free of the data's units: a change of units multiplies every run's
        predictive density by one constant, which the posterior's
        normalization takes out. A prior given is in the data's units.
    ``mean_prior``, ``mean_precision_prior``, ``degrees_of_freedom_prior``, ``covariance_prior``
        The conjugate prior, under scikit-learn's ``BayesianGaussianMixture``
        names:

        .. list-table::
           :header-rows: 1
           :widths: 20 50 30

           * - parameter
             - what it is
             - default
           * - ``mean_prior``
             - ``mu_0``, one value per feature
             - the first ``warm_rows`` rows' mean
           * - ``mean_precision_prior``
             - the weight of that mean in rows, finite and above 0
             - 1.0
           * - ``degrees_of_freedom_prior``
             - the degrees of freedom, finite and above ``d + 1`` under
               ``"full"`` (where the predictive has a variance) and above 1
               under ``"diag"`` (where it has a mean)
             - the smallest integer that gives the prior's variance a mean:
               ``d + 2`` under ``"full"``, whose inverse-Wishart needs
               ``nu > d + 1``, and 3 under ``"diag"``, whose per-feature
               normal-inverse-gamma needs ``nu > 2`` whatever ``d``
           * - ``covariance_prior``
             - the prior scale of the variance as a list, in the data's units:
               one positive number ``[s]`` for ``s`` times the identity, or the
               ``d * d`` entries of a symmetric positive-definite matrix, row by
               row
             - the first ``warm_rows`` rows' covariance, its diagonal under
               ``"diag"``

        ``d`` is the number of features. ``mean_precision_prior`` and
        ``degrees_of_freedom_prior`` are in rows of the stream's mean weight.
        A prior in the data's units decides what counts as surprising, so the
        defaults are the data's own (*The prior* in Notes).
    ``robust_beta``
        Temper the ``"diag"`` predictive, so that a single outlying row is
        not read as a break: each row is weighted by ``(pi(x) / pi(mode)) **
        robust_beta`` in what the run learns and in the likelihood it passes
        on. Default none, which tempers nothing: a single 20-sigma row is
        then a break whose run carries the outlier
        in its mean. A larger power finds a real shift later, and past about
        0.2 mostly not at all. Finite and above 0, and refused beside
        ``"full"``. The tempered predictive is not free of the data's units,
        so center and scale each feature before a tempered run. In the
        query before the bank, make its z-score ``(pl.col("x") -
        rp.ops.ewm_mean("x", half_life=h)) / rp.ops.ewm_std("x", half_life=h)``
        with ``lf.rheon.with_windows``, which runs the window operators
        (Polars' ``with_columns`` cannot).
        *Tempering* in Notes gives the measurements.
    ``prune_below``, ``max_run``
        ``prune_below`` is the share of the mass below which a run is
        dropped (default ``1e-6``, at least 0 and below 1). ``max_run`` caps
        a run's length: the runs that reach ``max_run - 1`` rows are folded
        into the one carrying the most mass (default 10,000, at least 2). A break
        collapses the runs to a few dozen, but a stream that does not break
        spreads them over thousands of run lengths. There ``max_run`` is the
        bound on the runs kept and on a row's work (*Pruning and the capped
        run* in Notes).
    ``hazard_kind``, ``hazard_k``, ``hazard_p``
        ``hazard_kind = "constant"`` (the default) uses a geometric run-length
        hazard via ``hazard_rows``, ``hazard`` or ``hazard_col``.
        ``hazard_kind = "neg_binomial"`` uses a negative-binomial run-length
        distribution with integer shape ``hazard_k >= 1`` and probability
        ``0 < hazard_p <= 1`` per learned row (``hazard_k = 1`` recovers a
        constant per-row hazard of ``1 / hazard_p``). Refused beside
        ``hazard``, ``hazard_rows`` and ``hazard_col``; ``hazard_k`` and
        ``hazard_p`` are refused unless ``hazard_kind = "neg_binomial"``.
    ``obs_var``
        Known observation variance per feature (one positive number, or a list
        of length ``d``). Activates the exact per-feature normal-normal
        conjugate emission using ``mean_prior`` and ``covariance_prior`` as
        the prior mean and variance of the unknown mean. Refused beside
        ``covariance_type = "full"``, ``degrees_of_freedom_prior``,
        ``mean_precision_prior`` and ``robust_beta``.
    ``max_state_bytes``
        Runtime per-group byte cap for live runs (default 262,144, ceiling
        64 MiB). Warns once per group when live runs pass half the cap and
        stops atomically if a row would exceed it.
    ``warm_rows``
        The learned rows held to set the prior from, where ``mean_prior`` or
        ``covariance_prior`` is left out. Default ``d + 2``, one more than
        ``d + 1``, the smallest count whose covariance is positive definite in
        general. :func:`hmm` and :func:`kmeans` name theirs the same way. At
        least 2, and refused beside both priors, which leave nothing to set.
        Where the first rows are not representative and rows are to spare,
        pass a ``warm_rows`` of tens, all from one regime (*The prior* in
        Notes).
    ``warm_span``
        The warm-up over a span of the clock in place of ``warm_rows``: the
        learned rows from the first through the first row at least
        ``warm_span`` after it, once they are at least two. A number of the
        clock's units, or of rows where the spec has no clock, or a duration
        such as ``"1h"`` on a temporal clock; finite and above 0. The span is
        read off the rows' exact stamps, and a row of weight 0 is no row of
        it, so on a clock of one unit a row it holds ``warm_span + 1`` rows:
        the prior of ``warm_rows = warm_span + 1``, to the bit. The data
        decides how many rows that is, so they are held to 256 MiB as they
        come: past half of it a warning, once a group, and the row that would
        pass it stops the run with an error naming the group and the bytes.
        Refused beside ``warm_rows`` and beside both priors.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. Every field but ``p_change`` is read before the
    row is learned. ``p_change`` is the posterior once this row's evidence
    is in.
    `docs/OUTPUTS.md#bocpd
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#bocpd>`_
    gives each field's type and when it is null:

    ``p_change``
        ``P(r_t <= 1)`` given this row, a posterior probability: the alarm,
        spiky and as big as the break is against the prior's scale. A group's
        first row is withheld under the default ``min_weight``. With both
        priors given and ``min_weight=0`` it reads 1 however the row looks
        (*Reading p_change and run_mode* in Notes).
    ``run_mode``
        The most likely run length before the row, so ``t - run_mode`` is
        the row the current run began on. It dates a break more reliably
        than ``p_change`` flags one.
    ``run_mean``
        The posterior mean run length before the row.
    ``pred_<f>``
        The pre-row predictive mean of each feature, mixed over runs as
        ``m* + sum_r p_r (m_r - m*)``, ``m*`` the most probable run's mean,
        so a feature at a level keeps the precision of its spread.
    ``loglik``
        The row's log predictive density under that mixture.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        Always null, since nothing decays.
    ``withheld_reason``
        Why the row's fields were withheld, null where they were not:
        ``below_min_weight`` on a group's first row under the default
        ``min_weight``, which takes precedence, then ``below_warm_rows`` on
        the rest of the rows held to set the prior from.

    A row whose predictive cannot be evaluated reports nulls and is counted
    in :meth:`rheon.polars.ModelBank.solve_failures`. It leaves the
    posterior where it stands, except for ``hazard``'s chance of a break
    over its step, which time passing applies.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`), ``TypeError`` for
    ``targets``, which this model does not take, and ``ValueError`` naming
    the problem for:

    - an empty ``features``, any ``half_life`` or ``lam``, and a
      ``covariance_type`` other than ``"full"`` or ``"diag"``;
    - a ``hazard_rows`` of 1 or less or not finite; a ``hazard`` at 0 or
      below or not finite, a duration ``hazard`` with no clock or beside a
      plain-number clock parameter, and a number ``hazard`` on a temporal
      clock or beside a duration; ``hazard`` beside ``hazard_rows`` or
      ``hazard_col``;
    - a ``mean_prior`` of the wrong length or holding a value that is not
      finite, a ``covariance_prior`` holding a value that is not finite, a
      ``mean_precision_prior`` not
      finite or not above 0, a ``degrees_of_freedom_prior`` not finite or
      at or below its floor, and a ``covariance_prior`` that is neither
      ``[s]`` nor ``d * d`` entries, or is not positive, symmetric and
      positive definite (checked whole, even under ``"diag"``, which reads
      its diagonal);
    - a ``robust_beta`` beside ``covariance_type="full"``, and one at 0,
      below it or not finite;
    - a ``prune_below`` outside ``[0, 1)``, a ``max_run`` below 2, a
      ``warm_rows`` below 2 or beside both priors, and held rows past
      256 MiB;
    - a switch of a diagnostic that reads residuals;

    and, from a call that learns or predicts, ``ValueError`` for a
    ``hazard_col`` value of 1 or less, naming the row.

    .. rubric:: The update

    Adams and MacKay's Algorithm 1, in log space. ``r`` is a run length,
    ``H = 1 / hazard_rows`` the chance of a break after the row, and ``pi_r`` run
    ``r``'s posterior predictive for this row:

    .. code-block:: text

        growth:      P(r_t = r+1, x_1:t) = P(r_t-1 = r, x_1:t-1) pi_r (1 - H)
        changepoint: P(r_t = 0,   x_1:t) = sum_r P(r_t-1 = r, x_1:t-1) pi_r H

    Under ``hazard``, the chance ``h = 1 - exp(-dt / tau)`` that a break
    fell inside the step belongs before the row. So ``h`` of every
    run's mass moves to the empty run before the row is read, and the
    recursion above runs with ``H = 0``, since no time has passed since the
    row:

    .. code-block:: text

        before the row:  P(r = 0) += h * sum_{r >= 1} P(r),   P(r) *= 1 - h for r >= 1

    Run ``r`` keeps the conjugate statistics of exactly the ``r`` rows that
    hypothesis says preceded this one in the run. So run 0 holds none, and
    its predictive is the prior's, which is what makes "a new run starts
    here" something the data can vote on. A row of weight ``w`` is learned
    at ``w / w_bar``, ``w_bar`` the mean weight of the rows learned, this one
    included, in its run's statistics and in the likelihood it passes the
    recursion. So a heavier row is more evidence of both kinds, and a
    constant multiple of every weight changes nothing. What the row reports
    is read as a row of the mean weight, which is all ``predict`` can know.

    .. rubric:: Example

    This code runs the model over a level that steps up by four standard
    deviations at row 600, and prints the row the current run began on,
    before and after the step:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(7)
        x = rng.standard_normal(1_000)
        x[600:] += 4.0                                # the level steps up at row 600
        pl.DataFrame({"x": x}).write_parquet("step.parquet")

        b = rp.spec.bocpd("regime", features=["x"], hazard_rows=250.0)
        out = pl.scan_parquet("step.parquet").rheon.fit_predict([b]).collect()
        out = out.unnest("regime")
        for row in [599, 650, 999]:
            began = row - out["run_mode"][row]        # the row the current run began on
            print(f"row {row}: the run began on row {began}")

    .. code-block:: text

        row 599: the run began on row 0
        row 650: the run began on row 600
        row 999: the run began on row 600

    .. rubric:: See also

    - :func:`corrchange`, tests for a change in the correlation of two
      features.
    - :func:`hmm`, a fixed number of regimes the stream moves between.
    - `docs/REGIMES.md
      <https://github.com/rheon-dev/rheon/blob/main/docs/REGIMES.md>`_,
      the models of regimes and changes, side by side.

    .. rubric:: Notes

    **Reading p_change and run_mode.** Under ``hazard_rows``,
    ``p_change`` is ``P(r <= 1)`` and not ``P(r = 0)``: the changepoint
    branch and the growth branch share the same predictive, which makes the
    normalized mass at ``r = 0`` exactly ``H`` on every row whatever the
    data. Under ``hazard``, the chance of a break is applied
    before the row, so no mass sits at ``r = 0`` after it, and ``p_change``
    is the chance that this row began a run. On regular steps of ``dt`` the
    two forms read every row against the same posterior, and ``p_rows = H +
    (1 - H) * p_clock`` with ``H = 1 - exp(-dt / tau)``: about ``H`` apart
    on a quiet row. There, under ``hazard``, a row whose clock step
    is 0 (a stamp shared with the last row the model did not skip) reports
    0, apart from a run's first row: a group's first row, and its first
    after a reset. With ``mean_prior`` and ``covariance_prior`` both
    given, the default ``min_weight`` withholds such a row, and with
    ``min_weight=0`` it reports a ``p_change`` of 1 there. Unless both are
    given, the first ``warm_rows`` rows report null.

    ``run_mode`` dates a break more reliably than ``p_change`` flags one. A
    break the posterior accepts a row or two late shows as a fall in
    ``run_mode``, not in ``p_change``. Over six seeds of a 5-sigma step at row 500, at the defaults,
    ``run_mode`` fell to 1 or 2 within two rows of the break on every seed,
    while the largest ``p_change`` near it ranged from 0.021 to 0.749.

    .. list-table::
       :header-rows: 1
       :widths: 44 30 26

       * - the break
         - ``p_change``
         - the run length
       * - the spread steps tenfold, under a prior of the rows' scale
         - 0.83 on the row of the break
         - found within a few rows, dated to the right row
       * - a four-sigma mean shift under a diffuse prior
         - barely lifts
         - the same
       * - a change in correlation alone
         - never moves
         - never found under ``"diag"``; under ``"full"``, found a median
           of 121 rows after the break

    **A clock hazard.** A step of 0 has no chance of a break: a row that
    shares the previous row's stamp cannot begin a run. A row of weight 0
    counts in no clock, so the next row's step runs from the last row that
    counted, one step of the time since. Without a clock column each row of
    positive weight is a step of 1. Under ``prune_below = 0`` each step of
    0 leaves one run of no mass among the runs kept, until ``max_run``
    folds it.

    **Tempering.** Untempered, a single 20-sigma row reads ``p_change``
    0.91, and its run carries the outlier in its mean. Tempered, an
    outlying row is atypical under every run,
    every tempered likelihood is about 1, and nothing moves. A whole new
    regime is a run of individually forgiven rows, which is why a larger
    power finds a real shift later. Over 50 streams with a four-sigma shift
    at ``hazard_rows=250``, a shift counts as found once ``run_mode`` dates it to
    within 5 rows, inside 150 rows:

    .. list-table::
       :header-rows: 1
       :widths: 20 80

       * - ``robust_beta``
         - the shifts found
       * - 0.1
         - all 50, the worst 6 rows late, while a 20-sigma outlier left
           ``p_change`` at its floor
       * - 0.15
         - all 50, the worst 16 rows late
       * - 0.2
         - 38, the worst 120 rows late
       * - 0.3
         - 1
       * - 0.5
         - none

    The tempered predictive's
    message ``pi ** w`` carries the constant a change of units multiplies
    the density by as ``c ** w``, and ``w`` differs from run to run, so the
    runs' odds move with the units. At a level of ``1e4`` with a spread of
    ``1e-2``, ``p_change`` moved by up to 0.86 against the same stream at
    level 0 and spread 1. ``robust_beta`` is a power on a ratio of two
    densities, and has no units.

    **The prior.** Too wide a prior, and the model goes quiet, because no
    row is surprising under a predictive that wide. Too narrow or
    off-center, and every run that begins on a row has no density for it.
    With ``mean_prior`` or ``covariance_prior`` left out, the first
    ``warm_rows`` learned rows report null and are held. Their weighted mean
    and covariance (``numpy.cov``'s with ``aweights``, the sample covariance
    at equal weights) set what was left out, and then they are read, in
    order, as rows of a model that had that prior from the start. At the
    default ``degrees_of_freedom_prior`` the prior's mean variance is then
    that covariance. A feature that does not move over those rows has no
    variance to give. It takes ``2**-52 * max(mean**2, 1)``, the spread of a
    value known to ``2**-26`` of its level, since a predictive of no spread
    has no density. Under ``"full"`` a covariance that is not positive
    definite, two features that are one, gives its diagonal. With both
    given there is no warm-up. ``degrees_of_freedom_prior`` and
    ``covariance_prior`` are ``2a`` and ``2b`` in the gamma
    parametrization, which is how Adams and MacKay give their own finance
    example (``a = 1``, ``b = 1e-4``, ``hazard_rows = 250``: in this model's
    terms ``degrees_of_freedom_prior=2.0`` and ``covariance_prior=[2e-4]``).

    The prior is set once from the ``warm_rows`` rows and kept for the
    stream's life, so at ``d + 2`` rows its scale rests on ``d + 1``
    degrees of freedom. For one feature, a 3-row variance lies between 0.05
    and 3 times the truth nine times in ten. A warm-up that straddles a
    regime change sets a wide prior: a variance of 37 across a 10-sigma
    step, against 0.53 from rows inside one regime.

    **Pruning and the capped run.** A row takes ``O(runs * d)`` under
    ``covariance_type="diag"``, tempered or not, and ``O(runs * d³)`` under
    ``"full"``, which factors each run's scale. The run vector would grow by
    one every row, so runs below ``prune_below`` of the mass are dropped.
    The runs that reach ``max_run - 1`` rows are folded into the one of them
    carrying the most mass, the oldest among equals, which takes their mass
    and the tail's place. Its statistics are scaled back to ``max_run - 1``
    rows' weight, as a decay would scale them: the weight and scatter
    shrink, the mean stays. No run kept is longer than ``max_run - 1``
    rows, so ``run_mode`` saturates one below it, and the runs kept, at most
    ``max_run`` lengths, bound the work a row takes. The capped run is an
    exponentially weighted run, scaled by ``(max_run - 1) / max_run`` each
    row. Its mean steps by ``1 / max_run`` of the row's deviation. Its
    half-life is ``ln 2 / ln(max_run / (max_run - 1))``, about
    ``0.69 * (max_run - 1)`` rows, 6,900 at the default. The run that
    dominates a quiet stream keeps its memory at the cap however many
    younger runs ``prune_below`` drops. After a break, the post-break run
    takes its place once it reaches the cap.

    **One feature is the slow case.** A row's work is the number of live
    runs times ``d``, and with one feature a row carries little evidence
    about the run length, so the posterior stays spread over many run
    lengths and many keep more than ``prune_below`` of the mass. On 60,000
    independent standard normal rows, with the defaults, one thread of an
    Apple M4 Pro, a row took about 120 µs with one feature, 3.5 µs with two
    and 1.4 µs with four. The default does not change with the number of
    features. A one-feature stream that needs speed sets ``prune_below``
    larger (``1e-4`` took 4 µs a row and ``1e-3`` 0.4 µs on those rows) or
    ``max_run`` smaller; either drops runs, so ``p_change`` and ``run_mode``
    move.
    """
    model: dict[str, Any] = {
        "type": "bocpd",
        "hazard": hazard,
        "hazard_col": hazard_col,
        "covariance_type": covariance_type,
        "mean_prior": mean_prior,
        "mean_precision_prior": mean_precision_prior,
        "degrees_of_freedom_prior": degrees_of_freedom_prior,
        "covariance_prior": covariance_prior,
        "robust_beta": robust_beta,
        "prune_below": prune_below,
        "max_run": max_run,
        "warm_rows": warm_rows,
    }
    # Written only when given, as the Rust spec leaves them out when absent.
    if hazard_kind != "constant":
        model["hazard_kind"] = hazard_kind
    for key, val in (
        ("warm_span", warm_span),
        ("hazard_rows", hazard_rows),
        ("hazard_k", hazard_k),
        ("hazard_p", hazard_p),
        ("obs_var", obs_var),
        ("max_state_bytes", max_state_bytes),
    ):
        if val is not None:
            model[key] = val
    mirror = _mirror_target(name, "bocpd", features, common, "its change points are in")
    targets = [hazard_col] if hazard_col is not None else mirror
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def corrchange(
    name: str,
    *,
    features: list[str],
    kind: Literal["monitor", "sequential", "window"] = "monitor",
    window_rows: int | None = None,
    alpha: float = 0.05,
    alpha_adjust: Literal["bonferroni", "holm", "none"] = "bonferroni",
    bandwidth: int | None = None,
    scalar: bool = False,
    crit: float | None = None,
    n_perm: int | None = None,
    permute_every_rows: int | None = None,
    perm_block: int | None = None,
    norm: Literal["l1", "linf"] = "l1",
    seed: int | None = None,
    change_action: Literal["flag", "reset"] = "flag",
    monitor_rows: int | None = None,
    boundary_gamma: float | None = None,
    permute_every: float | Duration | None = None,
    window_size: float | Duration | None = None,
    monitor_span: float | Duration | None = None,
    emit_n_flagged: bool = False,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Tests of whether the correlations between the feature columns have changed, by
    three kinds of test.

    Use ``kind="monitor"`` to ask whether the correlations were constant over
    each span of rows, ``kind="sequential"`` to test each row as it arrives
    against a history taken as stable, and ``kind="window"`` to ask how big
    the change between two adjacent windows is. Use :func:`bocpd` for the
    age of the current regime of the features' levels and spreads, and
    :func:`deco` to follow an equicorrelation that moves.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`rheon.spec`. This model takes ``features``, at least two,
    and no ``targets``. Nothing decays but the optional standardizer of
    ``scalar`` and ``weight_sum`` beside it, so ``half_life`` and ``lam``
    are refused except under ``scalar``. A row's positive weight reaches only
    ``weight_sum`` and that standardizer: the span, the windows and the
    history take each row unweighted. ``fit_intercept`` has nothing to act
    on, as there is no regression, and no diagnostic that reads residuals
    applies. A parameter that belongs to another kind is refused, naming the
    kinds it applies to. This model's own:

    ``kind``
        ``"monitor"`` (the default), ``"sequential"`` or ``"window"``.
    ``window_rows``
        A count of rows, required by every kind unless ``window_size`` is
        given: the span ``"monitor"`` tests (at least 8), the history
        ``"sequential"`` monitors against (at least 8), or each of
        ``"window"``'s two windows (at least 3).
    ``window_size``
        ``window_rows`` on the clock, in its place: a number of the clock's
        units (rows where the spec has no clock) or a duration such as
        ``"5m"`` on a temporal clock, above 0. A ``"monitor"`` span and a
        ``"sequential"`` history begin at their first row and end at the
        first row at least ``window_size`` after it, which is their last; on
        a clock of one unit a row, ``window_size = n`` is ``window_rows = n +
        1``, to the bit. ``"window"``'s two windows are the rows of ``(t -
        2 window_size, t - window_size]`` and ``(t - window_size, t]``, as
        Polars' ``rolling_*_by`` reads a window, and reports begin once the
        stream has run ``2 window_size``. Each span, history or pair of
        windows is tested at the rows it holds, with no least count of rows
        (*Spans on the clock* in Notes). A row of weight 0 counts in no clock.
    ``monitor_span``
        ``"sequential"`` under ``window_size``: the monitoring period on the
        clock, in place of ``monitor_rows``. Default ``window_size``.
    ``max_state_bytes``
        Under ``window_size``: the bytes a group's state may hold, since how
        many rows a span of the clock holds is the data's to say. Default 256
        KB, at most 64 MiB. Past half of it the bank warns once; a row that
        would pass it is not learned, and the bank refuses the run.
    ``alpha``, ``alpha_adjust``
        The level, in ``(0, 1)`` (default 0.05), and, under ``"monitor"`` and
        ``"sequential"``, how it is spread over the pairs: ``"bonferroni"``,
        the default, ``alpha / npairs``; ``"holm"``, Holm-Bonferroni step-down
        over the ordered pairs; or ``"none"``, ``alpha`` on each pair.
        Under ``"window"`` one permutation statistic covers every pair at once,
        so there is nothing to spread, and an ``alpha_adjust`` other than the
        default is refused. Under ``"sequential"`` without ``crit`` each pair's
        share is refused below ``5e-11``: the critical value is solved from the
        boundary's law, and in a smaller tail the solve's quantile is off by
        more than the 0.03 Wied and Galeano's own table is good to.
    ``emit_n_flagged``
        When ``True``, emit ``n_flagged`` (``i64``), the number of pairs
        flagged at the adjusted level (0 or 1 under ``scalar=True`` or
        ``kind="window"``; null when no report is due or when withheld).
        Default ``False``.
    ``bandwidth``
        ``"monitor"`` and ``"sequential"``: overrides the Bartlett bandwidth,
        ``floor(ln T)`` for a span or history of ``T`` rows. Lag ``l`` is
        weighted ``1 - l / bandwidth`` for ``l < bandwidth``, so at 1 only lag
        0 is left, and 0 is refused. At most ``window_rows``: a span of
        ``window_rows`` rows has no lag past ``window_rows - 1``, so a larger
        bandwidth adds no lag. It only flattens the weights toward 1, where
        the centered lags sum to 0, and the long-run variance falls toward 0
        with them.
    ``scalar``
        ``"monitor"`` and ``"sequential"``: in place of every pair, test the
        mean of ``u``, the equicorrelation of the standardized row
        (:func:`deco`'s ``u``), with the Bartlett long-run standard deviation
        of ``u`` in place of ``D``: one statistic however many columns there
        are.
        ``half_life``/``lam`` parametrize that standardizer, and
        ``weight_sum`` with it, and are accepted only here. Left out, nothing
        decays. Default ``False``.
    ``crit``
        ``"window"``: a fixed threshold, in place of the permutation quantile.
        ``"sequential"``: replaces Wied & Galeano's critical value. Finite
        and above 0. Default none.
    ``n_perm``, ``permute_every_rows``, ``perm_block``
        ``"window"``'s permutation quantile: ``n_perm`` draws (default 200, at most
        2^20, and without ``crit`` at least 20 and at least ``ceil(1 / alpha) -
        1``), redrawn every ``permute_every_rows`` rows (default ``window_rows``, at
        least 1), in blocks of ``perm_block`` rows (default 1, from 1 to
        ``window_rows``; at least 1 under ``window_size``). A block longer than
        a tenth of a window makes the held value too small for later
        windows: *The window test* in Notes gives the cost, the size and the
        blocks.
    ``permute_every``
        ``"window"``: the same redraw on the clock, every this long as well
        as every ``permute_every_rows`` reports, whichever comes first; given
        alone, the count beside it is none. A number of the clock's units
        (rows where the spec has no clock) or a duration such as ``"10m"``
        on a temporal clock, at least 0, ``0`` every report. On a clock of
        one unit a row, ``permute_every = n`` is ``permute_every_rows = n``,
        to the bit. Under ``window_size`` its default is ``window_size``.
    ``norm``
        ``"window"``: ``"l1"`` (the default) or ``"linf"``.
    ``seed``
        ``"window"``: seeds the permutation draws, at least 0, so two runs
        with the same seed report the same critical values. Default 0
        (``None`` is 0), and nothing draws from the system's entropy. The
        critical values a seed gives hold within one release of the package:
        a release may draw its permutations another way, from the same
        distribution, and a seed's critical value then moves by a few percent.
    ``change_action``
        ``"window"``: what a flag does, in the words ``drift_action`` takes:
        ``"flag"`` (the default) flags and carries on, and ``"reset"``
        empties the windows and starts over. A ``"monitor"`` span and a
        ``"sequential"`` cycle end at their flag already, so ``"reset"`` is
        refused there.
    ``monitor_rows``
        ``"sequential"``: the rows monitored after each history, the paper's
        ``floor(m T)``. Default ``window_rows`` (``T = 1``). At least 4, since a
        pair's monitored correlation carries no information before its fourth
        row (*The sequential test* in Notes), and at least 2 under ``scalar``.
    ``boundary_gamma``
        ``"sequential"``: the boundary's exponent, ``0 <= boundary_gamma <=
        0.49``. Default 0, the straight boundary ``1 + b``. Above 0 a
        change soon after the history is caught sooner and a late one later
        (*Choosing boundary_gamma* in Notes). At 0.5 the
        boundary would be crossed with probability 1, and the solve behind
        the critical value costs time in proportion to ``1 / (1/2 -
        boundary_gamma)`` on the way there, so a value above 0.49 is refused.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. ``stat``, ``crit``, ``flag`` and ``since_flag``
    are null except on a row where a statistic is due, and
    ``since_change`` except on a flag. The statistic reported on a row
    includes that row, and is reported before the row is learned. So a
    zero-weight row is reported as if it would be learned. It then does not
    enter the span, the history or the monitoring period, does not advance
    ``since_flag`` for the rows after it, and if it flags, it resets neither
    ``since_flag`` nor, under ``change_action="reset"``, the windows. A
    clock break (a step ``gap_cap`` shortened, or a session change)
    abandons the span, the windows or the cycle.
    `docs/OUTPUTS.md#corrchange
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#corrchange>`_
    gives each field's type and when it is null:

    ``stat``
        The test statistic: for the span, for the pair of windows, or for the
        monitored row against its boundary.
    ``crit``
        The critical value it is compared against: under ``"monitor"`` it
        reads the span's rows, under ``"sequential"`` with ``window_size`` the
        cycle's, and under ``"window"`` with ``window_size`` the report's two
        windows'.
    ``flag``
        True on the row where ``stat`` crossed ``crit``. Under ``"monitor"``,
        null on a span past ``crit`` whose two columns' scale ratio moved
        inside it, which ``withheld_reason`` names ``scale_ratio_moved`` (*The
        monitor test* in Notes).
    ``since_flag``
        Learned rows since the last flag.
    ``since_change``
        On a flag, the rows since the change it dates, counted through the
        flag's row from the first changed one. That row is the one after the
        maximum of the span's CUSUM (``"monitor"``), the
        one the paper's Eq. 8 gives (``"sequential"``), or the second
        window's first (``"window"``). Null otherwise. In the pairwise test
        (without ``scalar``), it is null where either feature of the flagged
        pair is constant over the monitored rows before a ``"sequential"``
        flag, since a correlation then has no spread to divide by.
    ``n_flagged``
        Under ``emit_n_flagged=True``, the count of pairs flagged at the
        adjusted level (``i64``); null when no report is due or withheld.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row, null where nothing decays.
    ``withheld_reason``
        Why the row's fields were withheld: null on every row at the default
        ``min_weight``, ``below_min_weight`` while ``weight_sum`` is below a
        ``min_weight`` given, and ``scale_ratio_moved`` where a ``"monitor"``
        flag is withheld.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`), ``TypeError`` for
    ``targets``, which this model does not take, and ``ValueError`` naming
    the problem for:

    - fewer than two features, and an ``alpha`` outside ``(0, 1)``;
    - a ``kind``, ``alpha_adjust``, ``norm`` or ``change_action`` not one
      of its words;
    - ``window_rows`` and ``window_size`` both left out, both given, or a
      ``window_rows`` below the kind's minimum, a ``window_size`` or
      ``monitor_span`` not above 0, and a ``monitor_span`` without
      ``window_size`` or beside ``monitor_rows``;
    - a ``max_state_bytes`` without ``window_size``, of 0 or above 64 MiB;
    - a parameter of another kind, and ``half_life`` or ``lam`` without
      ``scalar``;
    - under ``"sequential"`` without ``crit``, a share of ``alpha`` below
      ``5e-11`` a pair;
    - a ``bandwidth`` of 0 or above ``window_rows``, a ``crit`` not finite
      or not above 0, and an ``n_perm``, ``permute_every_rows``,
      ``perm_block`` or ``seed`` outside the range its parameter entry
      gives, an ``n_perm``
      among them too small to flag at ``alpha``;
    - a ``boundary_gamma`` outside ``[0, 0.49]``, and a ``monitor_rows``
      below 4 (below 2 under ``scalar``);
    - ``change_action="reset"`` under ``"monitor"`` or ``"sequential"``;
    - a switch of a diagnostic that reads residuals.

    .. rubric:: The update

    ``"monitor"`` is the constancy test of Wied, Krämer & Dehling (2012) in
    its linear form, run over consecutive spans of ``window_rows`` rows. At
    the last row of a span of ``T`` rows, per pair:

    .. code-block:: text

        xi_t = x_t * y_t - (rho_T / 2) * (x_t**2 + y_t**2)
        Q    = max_{1 <= j <= T} |sum_{t <= j} xi_t| / (sqrt(T) * D)
        crit = q(1 - alpha) - 0.5826 / sqrt(T)

    with ``x_t``, ``y_t`` the span's rows centered at its means and scaled by
    its standard deviations, ``rho_T`` their correlation, and ``D`` the
    Bartlett long-run standard deviation of ``xi`` (*The long-run deviation*
    in Notes). ``q`` is the Kolmogorov quantile, the law of ``sup|B|`` for a
    Brownian bridge ``B``. Over the pairs the statistic is the maximum, and
    the level is ``alpha / npairs`` under the default Bonferroni adjustment,
    or ``alpha`` under ``alpha_adjust="none"``. A pair flags where ``Q``
    passes ``crit`` and the same CUSUM of ``d_t = x_t**2 - y_t**2`` does
    not; where every pair past ``crit`` is one whose ``d`` passes it too,
    the flag is withheld.

    ``"sequential"`` is the monitoring procedure of Wied & Galeano (2013) on
    Fisher's scale. A cycle is ``window_rows`` learned rows of history
    (``m``), from which each pair's correlation ``rho_h`` and its ``D`` are
    read as for ``"monitor"``. Then the ``k``-th monitored row, for ``k`` up
    to ``monitor_rows`` (``K``), reports

    .. code-block:: text

        V_k  = ((k - 3) / sqrt(m - 3)) * |atanh(rho_k) - atanh(rho_h)| * (1 - rho_h**2) / D
        stat = max over pairs of V_k / w((k - 3) / (m - 3))
        w(b) = (1 + b) * (b / (1 + b)) ** boundary_gamma

    with ``rho_k`` the correlation of the ``k`` monitored rows so far, this
    one included, and flags where ``stat`` passes ``crit``. The cycle ends at
    a flag or after its ``K``-th row, and the next learned row starts a new
    history. Nothing is reported during the history, nor on the first three
    monitored rows, which carry no information on Fisher's scale. Under
    ``scalar`` the detector is the mean's, ``(k / sqrt(m)) * |mean_k(u) -
    mean_h(u)| / D_u`` against ``w(k / m)``, reported from the second
    monitored row.

    ``"window"`` reports ``norm(vech(R_pre - R_post))`` over two adjacent
    blocks of ``window_rows`` rows, ``R_pre`` and ``R_post`` their
    correlation matrices: how big the change is. The monitor asks instead
    whether the span was constant. It flags against ``crit``, or without one against a
    permutation quantile (*The window test* in Notes).

    .. rubric:: Example

    This code tests two features whose correlation jumps from 0 to 0.8 at
    row 1,100, over spans of 200 rows, and prints the span that flagged and
    the row its change was dated to:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(9)
        a, e = rng.standard_normal(2_000), rng.standard_normal(2_000)
        rho = np.where(np.arange(2_000) < 1_100, 0.0, 0.8)   # the jump at row 1,100
        b = rho * a + np.sqrt(1 - rho**2) * e
        pl.DataFrame({"a": a, "b": b}).write_parquet("pair.parquet")

        c = rp.spec.corrchange(
            "brk", features=["a", "b"],
            kind="monitor",                                     # constant in each span?
            window_rows=200,                                    # a verdict on each span's last row
        )
        out = pl.scan_parquet("pair.parquet").rheon.fit_predict([c]).collect()
        due = out.unnest("brk").with_row_index("row").filter(pl.col("stat").is_not_null())
        flags = due.filter("flag")
        print(f"{due.height} spans tested, {flags.height} flagged")
        for r in flags.iter_rows(named=True):
            first = r["row"] - r["since_change"] + 1          # the first changed row, as dated
            print(f"span ending on row {r['row']}: stat {r['stat']:.2f} against {r['crit']:.2f}")
            print(f"the change dated to row {first}")

    .. code-block:: text

        10 spans tested, 1 flagged
        span ending on row 1199: stat 2.51 against 1.32
        the change dated to row 1099

    .. rubric:: See also

    - :func:`bocpd`, a posterior over the age of the current regime.
    - :func:`deco`, an equicorrelation that moves with the stream.
    - `docs/REGIMES.md
      <https://github.com/rheon-dev/rheon/blob/main/docs/REGIMES.md>`_,
      the size and power of these tests, measured against their papers.

    .. rubric:: Notes

    **The long-run deviation.** ``D`` is the Bartlett long-run standard
    deviation of ``xi``, at bandwidth ``L = floor(ln T)`` with the paper's
    kernel, lag ``l`` at weight ``1 - l/L`` (their Appendix A.1). It is the
    paper's delta-method deviation of ``rho`` exactly: their five raw moments
    ``(x², y², x, y, xy)``, centered and mapped to ``(var_x, var_y, cov)``
    and then to ``rho``, give the series ``xi`` on the span centered at its
    means and scaled by its standard deviations, so the columns' level does
    not enter. A pair within rounding of ``|rho| = 1`` has no ``D`` and no
    verdict, as a constant column has none.

    **The monitor test.** The paper's statistic is ``max_j (j / sqrt(T)) *
    |rho_j - rho_T| / D``, with ``rho_j`` the correlation of the span's first
    ``j`` rows. Their proof reduces ``j (rho_j - rho_T)`` to the partial sum
    of ``xi`` plus a remainder, and the limit ``sup|B|`` is the law of the
    sum. The remainder is what over-rejects: ``rho_j`` of a few rows lives in
    ``[-1, 1]`` and is ``±1`` at two rows whatever the data, so at a
    correlation of 0.9 one span in seven was flagged by its second row. That
    statistic flagged 26% of stable spans of 100 rows at a correlation of
    0.9, and 41% of spans of 8 rows at 0 (the paper's own table has 0.142 at
    ``T = 200``). The partial sums form no correlation of a few rows, so they
    have no remainder to wait for: ``Q`` is valid at every ``T`` and every
    ``rho``. A span too short to carry a verdict cannot flag, with no least
    count written anywhere: ``xi`` sums to 0 over the span, so ``Q`` is at
    most ``max_j sqrt(j (T - j) / T)`` at bandwidth 1, under ``crit`` through
    ``T = 5``. The term ``0.5826 / sqrt(T)`` is ``-zeta(1/2) / sqrt(2 pi) /
    sqrt(T)``, how far a Brownian bridge's maximum over ``T`` points falls
    short of its supremum (Siegmund 1985; Broadie, Glasserman & Kou 1997),
    since ``Q`` is a maximum over the span's ``T`` rows. Share of stable spans
    flagged at a nominal 0.05, one pair, 10,000 spans a cell:

    .. list-table::
       :header-rows: 1
       :widths: 34 11 11 11 11 11 11

       * - stream
         - T = 8
         - 20
         - 50
         - 100
         - 200
         - 1000
       * - Gaussian, ``rho`` 0
         - 0.008
         - 0.013
         - 0.026
         - 0.037
         - 0.037
         - 0.043
       * - Gaussian, ``rho`` 0.6
         - 0.010
         - 0.017
         - 0.026
         - 0.031
         - 0.036
         - 0.043
       * - Gaussian, ``rho`` 0.9
         - 0.008
         - 0.016
         - 0.024
         - 0.036
         - 0.036
         - 0.043
       * - multivariate t, 5 df, ``rho`` 0.9
         - 0.006
         - 0.011
         - 0.015
         - 0.021
         - 0.023
         - 0.031
       * - one GARCH volatility, persistence 0.98, ``rho`` 0.6
         - 0.009
         - 0.017
         - 0.026
         - 0.035
         - 0.040
         - 0.044
       * - both standard deviations double mid-span, ``rho`` 0.6
         - 0.011
         - 0.016
         - 0.022
         - 0.027
         - 0.035
         - 0.046

    So the level is under nominal at short spans and heavy tails, and
    reaches it by 1000 rows. Power, a jump at the span's middle on Gaussian
    pairs, at ``T`` = 100 / 200 / 500: 0.19 / 0.42 / 0.83 from 0.5 to 0.7,
    0.54 / 0.87 / 0.94 from 0.9 to 0.7, 0.19 / 0.41 / 0.82 from 0 to 0.3. A
    jump in the first tenth of a span, or in its last, is found less often:
    0.18 and 0.15 of spans of 500 rows. The paper's statistic had about the
    same power once both are calibrated, and more for a jump near the span's
    start, for the same reason it over-rejected.

    What ``Q`` assumes is that the two columns' scales hold their ratio over
    the span. Both doubling together is a change of clock for the partial
    sums, which ``sup|B|`` does not see (the last row above). But with the
    correlation constant and the columns' scales at ``r_x`` and ``r_y`` times
    their span values, ``xi`` has mean ``-(rho/2)(r_x - r_y)²``: when one
    column's scale moves, ``Q`` leaves its null law, at ``rho`` 0.9 past
    ``crit`` on 93% of spans of 1000 rows whose one column doubles mid-span.
    The CUSUM of ``d = x² - y²`` has the same limit while the ratio holds,
    and grows with ``sqrt(T)`` when it moves, so it is the check: where it
    passes ``crit`` too, the flag is withheld. It costs a correlation jump
    about 5% of its flags, the check's own false alarms (0.04 / 0.06 of
    spans at ``T`` = 500 at the jumps above). On spans where one column's
    scale doubles at a correlation of 0.6, the flags withheld are 0.002 /
    0.014 / 0.037 / 0.056 / 0.12 at ``T`` = 20 / 50 / 100 / 200 / 1000, and
    the flags left are 0.015 / 0.020 / 0.005 / 0.001 / 0.000. Use
    ``"window"`` where the scales move and a verdict on every span matters:
    each of its windows' correlations is free of its own scale.

    A span reports on its last row only, and tests only its own rows. So a
    change is found at most ``window_rows`` rows late, and reliably only
    where it falls well inside a span. A jump in correlation from 0 to 0.8
    at row 1,100 of spans of 200 rows was found on 17 to 20 of 20 seeds;
    at rows 1,190 and 1,195, near the span's end, on 0 to 2 of 20 (each of
    three sets of 20 seeds).

    **The sequential test.** The critical value is Wied & Galeano's: with
    ``T`` the period's length in the detector's time, ``(K - 3) / (m - 3)``
    (``K / m`` under ``scalar``), ``crit = (T / (1 + T)) ** (1/2 -
    boundary_gamma) * q``. ``q`` is the ``1 - alpha`` quantile of
    ``sup_{0 < s <= 1} |W(s)| / s ** boundary_gamma`` for a Brownian motion
    ``W`` (their Eq. 7), at ``alpha / npairs`` under Bonferroni. At
    ``boundary_gamma = 0`` it comes from that law's series, 2.2414 at 5%, so
    ``crit`` is 1.5849 at ``T = 1``. Above 0 there is no series and the
    paper simulates. Here the law is solved as the diffusion it is, without
    simulation error. The values are within 0.03 of the paper's Table 1 and
    above it in 11 of its 12 cells, since a simulation on a grid misses
    crossings between its points. ``crit`` replaces the value.

    The detector is theirs with each side's correlation on Fisher's scale,
    at its own finite-sample variance: ``atanh(rho)`` of ``n`` rows has
    variance ``sigma_z² / (n - 3)``, so ``V_k`` has variance ``b + b²`` at
    ``b = (k - 3) / (m - 3)``, the paper's limit exactly and not only as
    ``m`` grows. Their own detector, ``(k / sqrt(m)) * |rho_k - rho_h| / D``
    from ``k = 2``, compared a correlation of two rows, ``±1`` whatever the
    data, with the history's: at ``boundary_gamma = 0.45`` and a correlation
    of 0.6, a quarter to a third of the cycles flagged in their first rows
    with nothing changed, at every history up to 1000 rows. Share of cycles
    ending in a flag with nothing changed, ``monitor_rows = window_rows``,
    one pair, 10,000 cycles a cell (5,000 at ``m = 1000``):

    .. list-table::
       :header-rows: 1
       :widths: 46 12 12 12 12

       * - stream, ``boundary_gamma``
         - m = 50
         - 100
         - 250
         - 1000
       * - Gaussian, ``rho`` 0, 0
         - 0.077
         - 0.063
         - 0.060
         - 0.051
       * - Gaussian, ``rho`` 0.6, 0.45
         - 0.071
         - 0.067
         - 0.057
         - 0.047
       * - Gaussian, ``rho`` 0.9, 0
         - 0.073
         - 0.067
         - 0.055
         - 0.059
       * - Gaussian, ``rho`` 0.9, 0.45
         - 0.065
         - 0.063
         - 0.059
         - 0.048
       * - multivariate t, 5 df, ``rho`` 0, 0
         - 0.121
         - 0.096
         - 0.083
         - 0.081
       * - multivariate t, 5 df, ``rho`` 0.9, 0.45
         - 0.093
         - 0.085
         - 0.066
         - 0.059
       * - GARCH, the paper's design, ``rho`` 0.5, 0
         - 0.083
         - 0.070
         - 0.063
         - 0.058
       * - GARCH, the paper's design, ``rho`` 0.5, 0.45
         - 0.074
         - 0.069
         - 0.059
         - 0.050
       * - one standard deviation doubles where the monitoring starts, 0.6, 0
         - 0.076
         - 0.060
         - 0.057
         - 0.053
       * - both double half way through the monitoring, 0.6, 0
         - 0.160
         - 0.160
         - 0.150
         - 0.156

    What is left above 0.05 is the history itself: a short history's
    ``rho_h`` and ``D``, and heavy tails' fourth moments, have not converged,
    which no treatment of the monitored rows can see. Take a history of a
    few hundred rows where the rows are heavy-tailed. A scale that changes
    between the history and the monitoring moves nothing, since each side's
    correlation is free of its own scale; a scale that changes inside the
    monitoring does (the last row), as the paper's stationarity assumption
    excludes. Power at ``m = 250``, flagged before the break / after it /
    median delay in rows: from 0.5 to 0.7 at monitored row 12, 0.000 / 0.890
    / 109 at ``boundary_gamma`` 0 and 0.014 / 0.770 / 93 at 0.45; from 0.9 to
    0.7 at row 62, 0.002 / 0.998 / 46 and 0.032 / 0.965 / 45; from 0 to 0.5
    at row 62, 0.001 / 0.982 / 73 and 0.037 / 0.911 / 79.

    On a flag, ``since_change`` dates the change as the paper's Eq. 8 does:
    ``k_hat = argmax_{j < tau} j * |rho_mon_j - rho_mon_{tau-1}|`` over the
    rows monitored before the flag, the history left out (they found it
    distorts the estimate). It reports ``tau - k_hat``, the rows from the
    first changed one through the flag's. They found it biased late for a
    change early in the period and early for one in its middle, both less as
    ``m`` and ``T`` grow.

    What it assumes, as the paper does: the history's correlations are
    constant (their Assumption 1, which ``kind = "monitor"`` over the same
    rows checks), the rows have finite fourth moments, and dependence fades
    (near-epoch dependence, which admits GARCH). Against ``"monitor"``, every
    row is tested and the baseline is fixed. So a change is flagged as soon
    as it is large enough, and a drift that ``"monitor"``'s spans would each
    absorb is measured against one level.

    **Choosing boundary_gamma.** Above 0 the boundary is lower at the start
    of the period, so a change soon after the history is caught sooner and a
    late one later: from 0.5 to 0.7 at monitored row 12, a median of 93 rows
    at 0.45 against 109 at 0, found on 0.77 of the cycles against 0.89. The
    size holds at every ``boundary_gamma`` (the table above). The paper's
    summary:

    .. list-table::
       :header-rows: 1
       :widths: 20 80

       * - ``boundary_gamma``
         - when
       * - 0
         - the pair is watched for a long time and false alarms are to be
           avoided, or a change is not expected soon after the history
       * - 0.45
         - to catch a change soon after it as fast as possible, false alarms
           accepted
       * - 0.25
         - a compromise

    **The window test.** Without ``crit`` the critical value is a
    permutation quantile: ``n_perm`` draws of the pooled rows shuffled
    between the two windows, redrawn every ``permute_every_rows`` rows. A
    report flags when fewer than ``floor(alpha * (n_perm + 1))`` draws reach
    its statistic. That is the standard permutation test, which counts the
    observed statistic as one more draw. On rows the null makes
    exchangeable, its rank among the ``n_perm + 1`` is uniform. So a test of
    a report against draws of its own windows has a size of that count over
    ``n_perm + 1``, at most ``alpha``: at alpha 0.05, 0.0476 at 20 draws and
    0.0498 at the default 200. The critical value is drawn from the windows
    after a report and held for ``permute_every_rows`` reports, so each
    later report is compared with draws of windows up to
    ``permute_every_rows - 1`` rows older than its own. Measured at alpha
    0.05, 200 draws and windows of 100 rows, the value held for 100
    reports, the share of reports flagged is ``alpha``: 0.049 on Gaussian
    rows at a correlation of 0 and 0.053 at 0.9, 0.049 under a GARCH
    volatility, 0.054 and 0.060 with both or one standard deviation
    doubling inside the windows. The test does not assume the scales hold.

    A stretch of reports is another matter. Two windows that slide by one
    row are almost the same windows, so a statistic above the quantile
    stays above it for a run of rows (5 to 7, about 7 runs in 1000 quiet
    rows at windows of 100). So at least one flag falls within a window's
    length of reports 0.33 of the time, within five windows' lengths 0.79 to
    0.82, and within ten 0.95 to 0.96, on Gaussian rows where nothing
    changed. A report says how
    large the change between its two windows is against its own null; use
    ``"sequential"`` to ask whether anything has changed since a history.

    The draws are in blocks of ``perm_block`` rows, so that serial
    dependence does not make the null too liberal: on levels whose lag-one
    correlation is 0.5, 0.124 of reports flag at blocks of 1 and 0.069 at
    blocks of 10 (windows of 100). A block holds back that liberality only
    while the windows hold many blocks. The value is an order statistic of
    the distinct splits of the pool's blocks into two windows (35 for 8
    blocks, 3 for 4), and held for later windows it is too small, more so
    the older it is. Share of reports flagged on independent rows, windows
    of 100, at ``perm_block`` 1 / 10 / 25 / 50: 0.049 / 0.058 / 0.083 /
    0.224, and about a half at ``perm_block = window_rows``, where the value
    is the one observed statistic. Keep ``perm_block`` to a tenth of
    ``window_rows`` or less. It is not a sign-flip null. Negating a whole row leaves
    every ``x x'`` and so every correlation matrix exactly where it was, so a
    sign-flip null has no spread at all. The quantile is drawn from the
    windows after a row is reported, so the first report once the windows
    fill has no critical value and cannot flag: ``crit`` is null there, and
    in force from the next row. A ``change_action="reset"`` empties the
    windows and the value in force with them, so the first report after the
    refill has none either.

    A redraw shuffles the ``2 * window_rows`` pooled rows ``n_perm`` times
    and reads both windows' correlations each time, so it costs ``n_perm *
    window_rows * p**2`` for ``p`` features. At the default
    ``permute_every_rows`` (``window_rows``), a redraw comes once a window's
    length of rows, that is ``n_perm * p**2`` a row, whatever the window, so
    the draws' cost per row does not grow with ``window_rows``. Each drawn
    value is held for those ``window_rows`` rows. A smaller
    ``permute_every_rows`` redraws more often, at more cost a row.

    **Spans on the clock.** ``window_size`` reads each span, history and
    window off the rows' stamps, so each holds the rows the data put in it
    and is tested at its own count: ``"monitor"``'s grid term and
    ``"sequential"``'s ``(K - 3) / (m - 3)`` read it, a cycle's period
    planned as ``K = floor(m * monitor_span / window_size)`` rows and ended at
    the first of those and its own span (a shorter period is crossed less
    often, so the size holds however the rows arrive). The window kind's
    permutation needs no equal windows: on exchangeable rows every
    assignment of the pooled rows to two windows of ``n1`` and ``n2`` is
    equally likely. Its held value is rescaled to each report's windows by
    ``sqrt((1/n1 + 1/n2) / (1/n1' + 1/n2'))``, ``n1'`` and ``n2'`` the
    windows it was drawn from, since a difference of two windows'
    correlations spreads as that root: drawn at 100 rows a window and held
    as the rate halves, unscaled it would turn 5% into 17%. A clock
    window's correlations are read whole, ``p**2`` times its rows a report,
    where ``window_rows`` merges block summaries.
    """
    model: dict[str, Any] = {
        "type": "corrchange",
        "kind": kind,
        "window_rows": window_rows,
        "alpha": alpha,
        "alpha_adjust": alpha_adjust,
        "bandwidth": bandwidth,
        "scalar": scalar,
        "crit": crit,
        "n_perm": n_perm,
        "permute_every_rows": permute_every_rows,
        "perm_block": perm_block,
        "norm": norm,
        "seed": seed,
        "change_action": change_action,
        "monitor_rows": monitor_rows,
        "boundary_gamma": boundary_gamma,
    }
    # Written only when given, as the Rust spec leaves them out when absent.
    for key, value in (
        ("permute_every", permute_every),
        ("window_size", window_size),
        ("monitor_span", monitor_span),
    ):
        if value is not None:
            model[key] = value
    if emit_n_flagged is not False:
        model["emit_n_flagged"] = emit_n_flagged
    targets = _mirror_target(name, "corrchange", features, common, "its test is of")
    return _common(name, model, targets=targets, features=features, **common)
