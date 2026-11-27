"""Online Hoeffding tree builders, :func:`hoeffding_tree_classifier` and
:func:`hoeffding_tree_regressor`: a part of :mod:`rheon.spec`, split from
``_spec.py`` to keep it under the repository's file-size cap. Import them
from :mod:`rheon.spec`.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Unpack

from rheon._kwargs import CommonKwargs
from rheon._spec import _SKIPPED_WHEN_ABSENT, TargetList, _checked, _common
from rheon._spec_words import quoted
from rheon._warnings import ReadinessWarning, _user_stacklevel

_DEFAULT_MAX_STATE_BYTES = 262_144
_WIDE_REGRESSOR_FEATURES = 20
_WIDE_REGRESSOR_SUGGESTED_BYTES = 1_048_576

_SKIPPED_WHEN_ABSENT["hoeffding_tree_classifier"] = frozenset(
    {
        "grace_period",
        "delta",
        "tau",
        "split_criterion",
        "leaf_prediction",
        "nb_threshold",
        "n_splits",
        "min_branch_fraction",
        "max_share_to_split",
        "merit_preprune",
        "max_depth",
        "max_state_bytes",
    }
)

_SKIPPED_WHEN_ABSENT["hoeffding_tree_regressor"] = frozenset(
    {
        "grace_period",
        "delta",
        "tau",
        "radius",
        "max_slots",
        "min_samples_split",
        "merit_preprune",
        "max_depth",
        "max_state_bytes",
    }
)


@_checked
def hoeffding_tree_classifier(
    name: str,
    *,
    features: list[str],
    label: str,
    classes: list[str],
    grace_period: float = 200.0,
    delta: float = 1e-7,
    tau: float = 0.05,
    split_criterion: Literal["info_gain", "gini", "hellinger"] = "info_gain",
    leaf_prediction: Literal["nba", "nb", "majority"] = "nba",
    nb_threshold: float = 0.0,
    n_splits: int = 10,
    min_branch_fraction: float = 0.01,
    max_share_to_split: float = 0.99,
    merit_preprune: bool = True,
    max_depth: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Online Hoeffding tree classifier.

    Fits an incremental Very Fast Decision Tree (VFDT) for streaming
    classification with nonlinear decision boundaries and feature interactions
    when linear discriminant boundaries (:func:`ew_class`) are too restrictive.
    Each active leaf tracks per-feature, per-class running Gaussian statistics (mean, variance,
    minimum, maximum) and splits once the Hoeffding bound guarantees that
    the best split beats the runner-up (or a tie falls below ``tau``).

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``, ``group``,
    ``weight`` and ``min_weight``, are in :mod:`rheon.spec`. A ``label``
    column takes the place of ``targets``. Neither ``half_life``, ``lam``
    nor ``min_settled_frac`` applies: Hoeffding-bound splits accumulate leaf
    statistics without exponential decay or hard windows. Row weights are
    frequency weights (an observation of weight ``w`` counts as ``w``
    observations toward ``grace_period``, the Hoeffding bound and leaf
    counts). A row of weight 0 is scored without changing the tree or any
    counter, and a row with a null label is scored without learning. No
    residual diagnostic or ``emit_feature_health`` applies.

    This model's own parameters:

    ``label``
        The column that holds each row's class, read as a key: any dtype
        with a string form, so ``["0", "1"]`` for an integer column and
        ``["true", "false"]`` for a boolean one.
    ``classes``
        Every value the label can hold, in the order the ``p_<class>``
        fields are written. Required, at least two, each named once and
        none ``""``. A value outside ``classes`` is refused, naming the row.
    ``grace_period``
        Minimum weight a leaf must accumulate between split attempts.
        Default ``200.0``, finite and above 0.
    ``delta``
        Hoeffding bound confidence level in ``(0, 1)``: smaller values
        demand stronger evidence before splitting. Default ``1e-7``.
    ``tau``
        Tie-breaking threshold in ``[0, 1]``: split when the Hoeffding
        bound falls below ``tau`` even if the top two candidates remain
        close. Default ``0.05``.
    ``split_criterion``
        Merit function used to score candidate splits:

        - ``"info_gain"`` (default): Shannon information gain, range
          ``log2(C)``.
        - ``"gini"``: Gini impurity reduction, range ``1.0``.
        - ``"hellinger"``: Hellinger distance (binary classification only,
          refused when ``len(classes) != 2``), range ``2 * sqrt(2)``.
    ``leaf_prediction``
        How each active leaf produces class probabilities:

        - ``"nba"`` (default): Naive Bayes Adaptive, tracking cumulative
          prequential accuracy of Naive Bayes versus majority class at the
          leaf and choosing whichever is ahead (ties go to Naive Bayes).
        - ``"nb"``: Gaussian Naive Bayes once the leaf's total weight
          reaches ``nb_threshold``, majority class before then.
        - ``"majority"``: leaf class proportions.

        Inactive (cap-deactivated) leaves always predict from their class
        counts.
    ``nb_threshold``
        Minimum leaf weight before ``leaf_prediction="nb"`` switches from
        majority class to Gaussian Naive Bayes. Default ``0.0``, finite and
        at least 0.
    ``n_splits``
        Number of evenly spaced candidate thresholds evaluated between each
        feature's observed minimum and maximum at a leaf. Default ``10``,
        at least 1.
    ``min_branch_fraction``
        Minimum weight share required on each side of a candidate split, in
        ``[0, 0.5)``. Default ``0.01``.
    ``max_share_to_split``
        Maximum majority-class share at which a leaf still attempts splits,
        in ``(0, 1]``. Default ``0.99``.
    ``merit_preprune``
        When ``True`` (default), include the no-split null candidate at
        merit ``0.0`` so a single uninformative feature cannot trigger a
        spurious split once the Hoeffding bound drops below ``tau``.
    ``max_depth``
        Optional maximum tree depth (``>= 1``). ``None`` (default) leaves
        depth unbounded up to the memory cap.
    ``max_state_bytes``
        Per-group packed state cap in bytes. Default ``262144`` (256 KB),
        ceiling ``67108864`` (64 MiB), refused below the bytes needed for
        one active root leaf. When splitting would exceed the cap, the tree
        stops growing and deactivates the lowest-promise leaves (dropping
        their Gaussian observers while continuing to update their class
        counts and predict from them), re-ranking leaves every 1,000
        learned rows.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields, all read before the row is learned.
    `docs/OUTPUTS.md#hoeffding_tree_classifier
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#hoeffding_tree_classifier>`_
    gives each field's type and when it is null:

    ``class``
        The class with the largest predicted probability (earliest in
        ``classes`` on a tie), an ``Enum`` of ``classes`` in their order.
        Null before ``min_weight`` and before any labeled row is learned.
    ``p_<class>``
        One per class in ``classes`` order: its predicted probability,
        summing to 1.
    ``weight_sum``
        Total weight accepted by the model before the row.
    ``settled_frac``
        Always null (no exponential decay).
    ``withheld_reason``
        Why the row's prediction was withheld (``"below_min_weight"`` until
        labeled weight reaches ``min_weight``), or null.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`), ``TypeError`` for
    ``targets`` (pass ``label`` instead), and ``ValueError`` for fewer than
    two ``classes``, duplicate or empty class names, ``split_criterion=
    "hellinger"`` with more than two classes, ``half_life``, ``lam`` or
    ``min_settled_frac``, out-of-range hyperparameters, ``label`` listed in
    ``features``, or ``max_state_bytes`` below one active root leaf or
    above 64 MiB.

    .. rubric:: The update

    Each row is routed from the root through internal threshold tests
    ``x[f] <= threshold`` to a leaf ``L`` and scored out-of-sample before
    updating. If the row has positive weight ``w`` and a non-null label
    ``c``:

    .. code-block:: text

        count[L, c] <- count[L, c] + w
        if L is active:
            update per-(feature, class) Gaussian observer (n, mean, M2, min, max)
            if total_weight(L) - last_attempt(L) >= grace_period:
                evaluate n_splits thresholds per feature via Gaussian CDF split
                eps = sqrt(R^2 * ln(1 / delta) / (2 * n_L))
                split if (best - second_best > eps) or (eps < tau)
                enforce max_state_bytes by ranking leaves on promise = n_L - max_c count[L, c]

    Packed state size per group is
    ``40 + 8 * C + 20 * B + (27 + 8 * C) * L + 40 * k * C * L_active``
    bytes for ``k`` features, ``C`` classes, ``B`` branches, ``L`` leaves
    and ``L_active`` active leaves.

    .. rubric:: Example

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(7)
        x0 = rng.standard_normal(4_000)
        x1 = rng.standard_normal(4_000)
        lab = np.where(x0 * x1 > 0, "pos", "neg")
        frame = pl.DataFrame({"x0": x0, "x1": x1, "lab": lab})

        tree = rp.spec.hoeffding_tree_classifier(
            "tree", features=["x0", "x1"], label="lab", classes=["neg", "pos"],
            grace_period=50.0, delta=1e-4,
        )
        out = rp.fit_predict(frame, [tree]).unnest("tree")
        acc = (out["class"].tail(1_000).cast(pl.String) == frame["lab"].tail(1_000)).mean()
        print(f"tail accuracy: {acc:.2f}")

    .. code-block:: text

        tail accuracy: 0.96

    .. rubric:: See also

    - :func:`hoeffding_tree_regressor`, the regression tree counterpart.
    - :func:`ew_class`, for exponentially weighted Gaussian classification
      with drift decay and hard windows.
    """
    if "targets" in common:
        msg = (
            f"spec {quoted(name)}: hoeffding_tree_classifier() takes no targets, since label "
            "names the column of its classes. Pass that column as label, and leave out targets."
        )
        raise TypeError(msg)
    model: dict[str, Any] = {
        "type": "hoeffding_tree_classifier",
        "classes": classes,
        "grace_period": None if grace_period == 200.0 else grace_period,
        "delta": None if delta == 1e-7 else delta,
        "tau": None if tau == 0.05 else tau,
        "split_criterion": None if split_criterion == "info_gain" else split_criterion,
        "leaf_prediction": None if leaf_prediction == "nba" else leaf_prediction,
        "nb_threshold": None if nb_threshold == 0.0 else nb_threshold,
        "n_splits": None if n_splits == 10 else n_splits,
        "min_branch_fraction": None if min_branch_fraction == 0.01 else min_branch_fraction,
        "max_share_to_split": None if max_share_to_split == 0.99 else max_share_to_split,
        "merit_preprune": None if merit_preprune else False,
        "max_depth": max_depth,
    }
    model = {k: v for k, v in model.items() if v is not None}
    return _common(name, model, targets=[label], features=list(features), **common)


@_checked
def hoeffding_tree_regressor(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    grace_period: float = 200.0,
    delta: float = 1e-7,
    tau: float = 0.05,
    radius: float = 0.25,
    max_slots: int = 32,
    min_samples_split: float = 5.0,
    merit_preprune: bool = True,
    max_depth: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Online Hoeffding tree for regression.

    Fits one FIMT-DD regression tree per target with bounded quantization
    observers, for streaming nonlinear regression with interaction effects and
    piecewise-constant predictions, supporting multiple targets and formula
    targets alongside residual diagnostics (``emit_sigma``, ``conformal``,
    ``emit_metrics``, ``emit_drift``, ``resid_quantiles``). Each target
    maintains its own tree; active leaves summarize each numeric feature in
    a bounded quantization observer and split using variance reduction and
    the Hoeffding bound on the runner-up-to-best ratio.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``, ``group``,
    ``weight``, ``min_weight`` and residual diagnostics, are in
    :mod:`rheon.spec`. Neither ``half_life``, ``lam`` nor
    ``min_settled_frac`` applies: Hoeffding-bound splits accumulate leaf
    statistics without exponential decay or hard windows. Row weights are
    frequency weights (an observation of weight ``w`` counts as ``w``
    observations toward ``grace_period``, ``min_samples_split`` and the
    Hoeffding bound). A row of weight 0 is scored without updating any
    tree or counter, and a null target skips learning on that target alone.
    Linear-only readings (``max_error_inflation``, ``emit_error_inflation``,
    ``emit_influence``, ``emit_se_coef``, ``emit_robust_se``,
    ``emit_support_coef``) are refused.

    This model's own parameters:

    ``grace_period``
        Minimum target weight accumulated in a leaf between split attempts.
        Default ``200.0``, finite and above 0.
    ``delta``
        Hoeffding bound confidence level in ``(0, 1)``. Default ``1e-7``.
    ``tau``
        Tie-breaking threshold in ``[0, 1]``: split when the Hoeffding
        bound falls below ``tau``. Default ``0.05``.
    ``radius``
        Quantization bucket width for each feature's observer, finite and
        above 0. Default ``0.25`` (best suited to roughly unit-scale
        features; scale ``radius`` with feature spread if needed).
    ``max_slots``
        Maximum number of quantization buckets retained per feature per
        active leaf (at least 2). Once full, new observations merge into
        the nearest existing bucket. Default ``32``.
    ``min_samples_split``
        Minimum weight required on each side of a candidate split (finite
        and above 0). Default ``5.0``.
    ``merit_preprune``
        When ``True`` (default), include the no-split null candidate at
        merit ``0.0`` so pure-noise leaves do not split on a tie.
    ``max_depth``
        Optional maximum tree depth (``>= 1``). ``None`` (default) leaves
        depth unbounded up to the memory cap.
    ``max_state_bytes``
        Per-group packed state cap in bytes, shared equally across targets
        when multiple targets are named. Default ``262144`` (256 KB),
        ceiling ``67108864`` (64 MiB), refused below the bytes required for
        one active root leaf per target. With 20 or more features, the
        default 256 KB cap leaves room for only a few active leaves at once
        and emits a :class:`~rheon.ReadinessWarning` suggesting
        ``max_state_bytes=1_048_576`` (1 MiB) or a smaller ``max_slots``.
        At the cap, the tree stops growing and deactivates the
        lowest-promise leaves (ranked by leaf sum of squared deviations
        ``S``), continuing to update leaf means and re-ranking every 1,000
        learned rows.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields, read before the row is learned.
    `docs/OUTPUTS.md#hoeffding_tree_regressor
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#hoeffding_tree_regressor>`_
    gives each field's type and when it is null:

    ``pred_<t>``
        Mean target value of the leaf the row routes to before learning.
        Null before the target's weight reaches ``min_weight`` or before
        its first non-null observation.
    ``resid_<t>``
        Target minus ``pred_<t>``.
    ``weight_sum``
        Total weight accepted by the model before the row.
    ``settled_frac``
        Always null (no exponential decay).
    ``withheld_reason``
        Why the row's prediction was withheld, or null.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`), and ``ValueError`` for
    ``half_life``, ``lam`` or ``min_settled_frac``, out-of-range
    hyperparameters (including ``max_slots < 2`` or ``radius <= 0``),
    linear-only diagnostics, or ``max_state_bytes`` below one active root
    leaf per target or above 64 MiB.

    .. rubric:: The update

    Each target ``j`` owns an independent tree. Before updating, ``pred_j``
    reads the mean ``mu_L`` of the leaf ``L`` reached by ``x``. For a
    non-null ``y_j`` with positive weight ``w``:

    .. code-block:: text

        update leaf Welford stats (n_L, mu_L, S_L) with (y_j, w)
        if L is active:
            update each feature's quantization observer (bucket = floor(x / radius))
            if n_L - last_attempt(L) >= grace_period:
                VR = var(S_obs) - (n_left/n_obs) var(left) - (n_right/n_obs) var(right)
                eps = sqrt(ln(1 / delta) / (2 * n_L))
                split if best_VR > 0 and (second_VR / best_VR < 1 - eps or eps < tau)
                enforce per-target cap by ranking leaves on promise = S_L

    Packed state size per target tree is
    ``40 + 20 * B + 35 * L + k * (2 + 40 * max_slots) * L_active`` bytes
    worst-case for ``k`` features, ``B`` branches, ``L`` leaves and
    ``L_active`` active leaves.

    .. rubric:: Example

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(11)
        x0 = rng.uniform(-2.0, 2.0, 4_000)
        x1 = rng.uniform(-2.0, 2.0, 4_000)
        y = np.where(x0 > 0, 2.0 * x1, -2.0 * x1) + 0.1 * rng.standard_normal(4_000)
        frame = pl.DataFrame({"x0": x0, "x1": x1, "y": y})

        reg = rp.spec.hoeffding_tree_regressor(
            "reg", targets=["y"], features=["x0", "x1"],
            grace_period=50.0, delta=1e-4,
        )
        out = rp.fit_predict(frame, [reg]).unnest("reg")
        mae = out["resid_y"].tail(1_000).abs().mean()
        print(f"tail MAE: {mae:.2f}")

    .. code-block:: text

        tail MAE: 0.23

    .. rubric:: See also

    - :func:`hoeffding_tree_classifier`, the classification tree
      counterpart.
    - :func:`isotonic`, for 1-D monotone nonparametric regression with
      decay or sliding windows.
    """
    model: dict[str, Any] = {
        "type": "hoeffding_tree_regressor",
        "grace_period": None if grace_period == 200.0 else grace_period,
        "delta": None if delta == 1e-7 else delta,
        "tau": None if tau == 0.05 else tau,
        "radius": None if radius == 0.25 else radius,
        "max_slots": None if max_slots == 32 else max_slots,
        "min_samples_split": None if min_samples_split == 5.0 else min_samples_split,
        "merit_preprune": None if merit_preprune else False,
        "max_depth": max_depth,
    }
    model = {k: v for k, v in model.items() if v is not None}
    spec = _common(name, model, targets=targets, features=list(features), **common)
    _msb = common.get("max_state_bytes")
    cap = _DEFAULT_MAX_STATE_BYTES if _msb is None else _msb
    if len(features) >= _WIDE_REGRESSOR_FEATURES and cap < _WIDE_REGRESSOR_SUGGESTED_BYTES:
        warnings.warn(
            ReadinessWarning(
                f"spec {quoted(name)}: hoeffding_tree_regressor has {len(features)} features "
                f"under max_state_bytes={cap}, which leaves little room for active quantization "
                f"observers. Raise max_state_bytes to at least {_WIDE_REGRESSOR_SUGGESTED_BYTES} "
                "(1 MiB) or lower max_slots so leaves can keep splitting."
            ),
            stacklevel=_user_stacklevel(),
        )
    return spec
