"""The streaming anomaly detector builders, :func:`half_space_trees` and
:func:`random_cut_forest`: a part of :mod:`rheon._spec` in a file of its own,
keeping that one under ``tests/test_repo_hygiene.py``'s 250 KB cap for a
source file. :mod:`rheon.spec` documents them with the rest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Unpack

from rheon._kwargs import CommonKwargs
from rheon._spec import _checked, _common, _mirror_target
from rheon._spec_words import _SKIPPED_WHEN_ABSENT, quoted


@_checked
def half_space_trees(
    name: str,
    *,
    features: list[str],
    n_trees: int = 10,
    height: int = 8,
    window_rows: int = 250,
    limits: Mapping[str, Sequence[float]] | None = None,
    seed: int = 0,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Streaming Half-Space Trees unsupervised anomaly detector.

    Use it for fast, constant-memory anomaly scoring over bounded numeric
    streams where each row's score is read from the previous tumbling
    reference window before the row is recorded in the latest window. Use
    :func:`random_cut_forest` when feature ranges are unknown in advance or
    when multi-step shingled subsequences matter.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``gap_cap``, ``min_weight``, ``weight`` and ``group``, are in
    :mod:`rheon.spec`. This model takes ``features`` and no ``targets``;
    ``half_life`` and ``lam`` are refused because Half-Space Trees forgets by
    tumbling its reference and latest windows every ``window_rows`` learned
    rows. This model's own:

    ``features``
        Numeric feature columns scored on every row.
    ``n_trees``
        Number of random half-space trees in the ensemble. Default ``10``.
    ``height``
        Depth of every full binary tree (``1`` to ``16``). Default ``8``.
    ``window_rows``
        Number of positive-weight rows in each tumbling window. After every
        ``window_rows`` learned rows, the latest-window mass profile becomes
        the reference profile and the latest profile resets to zero. Scores
        during the first window are withheld with ``below_warm_rows``.
        Default ``250``.
    ``limits``
        Optional mapping from feature column name to a ``(lo, hi)`` workspace
        interval with ``lo < hi``. Features omitted from ``limits`` default
        to ``(0.0, 1.0)``. Keys not present in ``features`` are refused.
    ``seed``
        Non-negative integer seed for the counter-based Philox4x64-10 draw
        stream that builds the trees. Default ``0``.
    ``max_state_bytes``
        Maximum serialized state size in bytes per group, at most 64 MiB.
        Default 2 MiB (``2_097_152``). Enforced eagerly from
        ``n_trees * (2 ** (height + 1) - 1)``.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields, read before the row is learned.
    `docs/OUTPUTS.md#half_space_trees
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#half_space_trees>`_
    gives each field's type and when it is null:

    ``score``
        Normalized anomaly score in ``[0, 1]`` where values near ``1`` denote
        sparse (anomalous) regions and values near ``0`` denote dense regions.
        Null during the first tumbling window (``below_warm_rows``) or below
        ``min_weight``.
    ``weight_sum``
        The accumulated weight before the row.
    ``settled_frac``
        Always null (Half-Space Trees forgets by tumbling windows, not decay).
    ``withheld_reason``
        Why the row's score was withheld (``below_warm_rows`` during the
        first window, ``below_min_weight`` below ``min_weight``), or null.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`), ``TypeError`` for ``targets``,
    and ``ValueError`` naming the problem for:

    - ``half_life`` or ``lam`` given;
    - ``n_trees < 1``, ``height`` outside ``1..=16``, or ``window_rows < 1``;
    - a ``limits`` key not in ``features``, an entry that is not a pair
      ``(lo, hi)``, or bounds with ``lo >= hi`` or non-finite values;
    - a forest whose mass-profile state exceeds ``max_state_bytes``;
    - ``coef_every`` or any residual-based diagnostic switch.

    .. rubric:: The update

    Each tree partitions a random workspace box around ``limits`` into a full
    binary tree of depth ``height``. For each node ``v``, ``r[v]`` holds the
    reference-window mass and ``l[v]`` holds the latest-window mass. Before
    updating on row ``x`` (once the first window has tumbled):

    .. code-block:: text

        ref_mass   = r[tree 0, root]
        size_limit = 0.1 * ref_mass
        raw_score  = sum_t r[t, stop_node(t, x)] * 2^depth(stop_node)
        score      = 1 - raw_score / (n_trees * ref_mass * (2^(height + 1) - 1))

    where traversal descends until a leaf or a node with ``r[v] < size_limit``.
    Then for every tree ``t``, ``l[v] += w`` along the root-to-leaf path of
    ``x``; every ``window_rows`` positive-weight rows, ``r <- l`` and
    ``l <- 0``.

    .. rubric:: Example

    .. code-block:: python

        import polars as pl
        import rheon.polars as rp

        xs = [0.5] * 30 + [0.99]
        df = pl.DataFrame({"x0": xs})
        spec = rp.spec.half_space_trees(
            "hst", features=["x0"], n_trees=5, height=4, window_rows=20, seed=7
        )
        out = rp.ModelBank([spec]).fit_predict(df)
        scores = out["hst"].struct.field("score").to_list()
        print(scores[0], scores[-1] > scores[-2])

    .. code-block:: text

        None True

    .. rubric:: See also

    - :func:`random_cut_forest`, adaptive-range anomaly detection with
      optional shingling and no required feature bounds.
    - :func:`ew_cov`, Mahalanobis distance under exponentially weighted
      Gaussian moments.

    .. rubric:: Notes

    **Weight scale invariance.** Because ``size_limit`` and the normalization
    denominator both scale with the reference window's total mass, scaling
    every row's weight by a positive constant leaves ``score`` invariant, and
    at unit weights the score matches the standard mass-profile definition
    bit-for-bit.
    """
    targets = _mirror_target(name, "half_space_trees", features, common, "it scores")
    if limits is not None:
        checked_limits: dict[str, list[float]] = {}
        for col, bounds in limits.items():
            if len(bounds) != 2:
                raise ValueError(
                    f"spec {quoted(name)}: limits[{quoted(col)}] must hold two bounds (lo, hi), "
                    f"but it has {len(bounds)}. Pass a pair (lo, hi) for each bounded feature."
                )
            checked_limits[col] = [float(bounds[0]), float(bounds[1])]
        limits_out: dict[str, list[float]] | None = checked_limits
    else:
        limits_out = None
    model: dict[str, Any] = {
        "type": "half_space_trees",
        "n_trees": n_trees,
        "height": height,
        "window_rows": window_rows,
        "limits": limits_out,
        "seed": seed,
    }
    skipped = _SKIPPED_WHEN_ABSENT["half_space_trees"]
    model = {k: v for k, v in model.items() if v is not None or k not in skipped}
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def random_cut_forest(
    name: str,
    *,
    features: list[str],
    n_trees: int = 40,
    tree_size: int = 256,
    shingle_size: int = 1,
    seed: int = 0,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Streaming Robust Random Cut Forest unsupervised anomaly detector.

    Use it to score anomalies on numeric streams without pre-specifying
    feature bounds, with optional shingling over consecutive rows
    (``shingle_size > 1``) to detect temporal pattern shifts. Each tree holds
    a bounded FIFO window of ``tree_size`` points and scores each incoming
    point by its average collusive displacement (CoDisp) before inserting it.
    Use :func:`half_space_trees` when feature ranges are known and higher
    throughput or fractional row weights are needed.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``gap_cap``, ``min_weight``, ``weight``, ``session`` and ``group``, are
    in :mod:`rheon.spec`. This model takes ``features`` and no ``targets``;
    ``half_life`` and ``lam`` are refused because each tree forgets by
    evicting its oldest point once it holds ``tree_size`` points. When a
    ``weight`` column is given, usable weights must be ``0`` (score-only) or
    ``1`` (insert). This model's own:

    ``features``
        Numeric feature columns scored on every row.
    ``n_trees``
        Number of random cut trees in the forest. Default ``40``.
    ``tree_size``
        Maximum number of points held in each tree's FIFO ring (at least 2).
        Default ``256``.
    ``shingle_size``
        Number of consecutive learned rows concatenated into each point
        (at least 1). Default ``1`` (no shingling). When ``shingle_size > 1``,
        a session change or a clock gap past ``gap_cap`` clears the shingle
        buffer and scores are withheld with ``below_warm_rows`` until
        ``shingle_size`` consecutive rows are present.
    ``seed``
        Non-negative integer seed for the counter-based Philox4x64-10 draw
        stream that samples cuts. Default ``0``.
    ``max_state_bytes``
        Maximum serialized state size in bytes per group, at most 64 MiB.
        Default 9 MiB (``9_437_184``). Checked eagerly from ``n_trees``,
        ``tree_size`` and the shingled dimension
        ``len(features) * shingle_size``.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields, read before the row is learned.
    `docs/OUTPUTS.md#random_cut_forest
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#random_cut_forest>`_
    gives each field's type and when it is null:

    ``score``
        Mean collusive displacement (CoDisp) across the forest, non-negative
        and unbounded above (larger values indicate stronger anomalies; ``0``
        before any point is inserted). Null while fewer than ``shingle_size``
        consecutive rows are available (``below_warm_rows``) or below
        ``min_weight``.
    ``weight_sum``
        The count of inserted points before the row.
    ``settled_frac``
        Always null (each tree forgets by FIFO eviction at ``tree_size``).
    ``withheld_reason``
        Why the row's score was withheld (``below_warm_rows`` while filling
        the shingle buffer, ``below_min_weight`` below ``min_weight``), or
        null.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`), ``TypeError`` for ``targets``,
    and ``ValueError`` naming the problem for:

    - ``half_life`` or ``lam`` given;
    - ``n_trees < 1``, ``tree_size < 2``, or ``shingle_size < 1``;
    - a configuration whose bounded forest state exceeds ``max_state_bytes``;
    - ``coef_every`` or any residual-based diagnostic switch.

    And, from a call that learns with a ``weight`` column, a compute error
    naming the row for any usable weight other than ``0`` or ``1``.

    .. rubric:: The update

    Each tree partitions its active points recursively by drawing a cut
    coordinate proportional to each dimension's bounding-box span and a cut
    value uniformly across that span, maintaining strict separation
    (``left <= cut < right``) on every internal node. Before inserting a
    point ``x`` into each tree, the score is the mean across trees of the
    collusive displacement ``x`` would induce if inserted without mutating
    the tree:

    .. code-block:: text

        CoDisp_t(x) = max_{u on path from x to root}  mass(sibling(u)) / mass(u)
        score(x)    = (1 / n_trees) * sum_t CoDisp_t(x)

    On a row of weight ``1``, each tree then evicts its oldest point if it
    already holds ``tree_size`` points and inserts ``x`` using the same
    counter-based cut draws.

    .. rubric:: Example

    .. code-block:: python

        import polars as pl
        import rheon.polars as rp

        xs = [0.0, 0.1, -0.1, 0.05, -0.05] * 6 + [25.0]
        df = pl.DataFrame({"x0": xs})
        spec = rp.spec.random_cut_forest(
            "rcf", features=["x0"], n_trees=10, tree_size=32, seed=7
        )
        out = rp.ModelBank([spec]).fit_predict(df)
        scores = out["rcf"].struct.field("score").to_list()
        print(scores[-1] > scores[-2])

    .. code-block:: text

        True

    .. rubric:: See also

    - :func:`half_space_trees`, fixed-grid mass-profile anomaly detection
      supporting arbitrary non-negative row weights.
    - :func:`ew_cov`, Mahalanobis distance under exponentially weighted
      Gaussian moments.

    .. rubric:: Notes

    **Evict-after-score and zero-weight rows.** Because each tree evaluates
    CoDisp against the current tree without evicting first and zero-weight
    rows do not advance the Philox counter or shingle buffer, scoring via
    :meth:`~rheon.polars.ModelBank.predict` matches the next step's score to
    the last bit and zero-weight rows leave the state byte-identical.
    """
    targets = _mirror_target(name, "random_cut_forest", features, common, "it scores")
    model: dict[str, Any] = {
        "type": "random_cut_forest",
        "n_trees": n_trees,
        "tree_size": tree_size,
        "shingle_size": shingle_size,
        "seed": seed,
    }
    skipped = _SKIPPED_WHEN_ABSENT["random_cut_forest"]
    model = {k: v for k, v in model.items() if v is not None or k not in skipped}
    return _common(name, model, targets=targets, features=features, **common)
