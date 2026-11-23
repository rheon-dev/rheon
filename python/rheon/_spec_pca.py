"""The streaming principal components' builder, :func:`pca`: a part of
:mod:`rheon.spec`, split from ``_spec.py`` to keep it under the repository's
file-size cap. Import it from :mod:`rheon.spec`.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._kwargs import CommonKwargs
from rheon._spec import _checked, _common, _mirror_target


@_checked
def pca(
    name: str,
    *,
    features: list[str],
    n_components: int | None = None,
    method: Literal["svd", "ccipca", "oja"] = "svd",
    learning_rate: float | None = None,
    max_state_bytes: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Principal components of the feature columns, learned one row at a time.

    Use it for the leading directions of a stream's columns, their variances
    and each row's scores, where the columns are too many for a ``k x k``
    covariance to be kept per group: its state grows with the number of
    components times the number of features, not with the features
    squared. Each row's components and scores are read before the row is
    learned, so a score can be a feature for that same row without leaking
    it. For a few hundred columns or fewer, :func:`ew_cov`'s ``pca`` gives
    the exact components of the same decayed covariance, and of a hard
    window too.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`rheon.spec`. ``half_life`` is in clock units: a number of the
    clock's units, or a duration on a temporal clock. This model reads
    ``features`` and takes no ``targets``. ``fit_intercept`` has nothing to
    act on. Nothing that reads residuals applies, and each such switch is
    refused by name: ``emit_sigma``, ``emit_metrics``, ``conformal``,
    ``emit_drift`` and the rest. ``coef_every`` and
    ``max_rows_between_coefs`` are refused, as there are no coefficients.
    ``min_weight`` defaults to 0: the components' own test (*Which
    components are written* in Notes) is the gate. This model's own:

    ``n_components``
        The components to write, at least 1 and at most the number of
        features. Default 2, or 1 for a single feature. One direction more
        is kept where there is room, as the last component's neighbor.
    ``method``
        How the components are learned (*The update*): ``"svd"`` (the
        default), the incremental singular value decomposition, exact while
        the data's rank is at most the directions kept; ``"ccipca"``,
        candid covariance-free incremental PCA; or ``"oja"``, Oja's rule.
        The last two are stochastic approximations that converge toward the
        components, at a rate set by the gaps between them.
    ``learning_rate``
        ``"oja"``'s gain on its rate, finite and above 0. Default 4.0.
        Refused under the other methods, which take no rate.
    ``max_state_bytes``
        The bytes a group's state may hold, above 0 and at most 64 MiB.
        Default 262,144 (256 KB). The state is
        ``8 * (2 + 3k + q*k + q)`` bytes for ``k`` features and ``q`` kept
        directions, fixed by the spec, so it is checked as the spec is
        built: 2,000 features and 12 components hold 256,120 bytes.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields, read before the row is learned.
    `docs/OUTPUTS.md#pca
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#pca>`_
    gives each field's type and when it is null. Per component ``j``,
    from 0:

    ``pc<j>_var``
        The variance along the component: its eigenvalue, times Bessel's
        correction at Kish's effective size, ``n / (n - 1)``, as
        scikit-learn's ``explained_variance_`` is at unit weights.
    ``pc<j>_share``
        Its share of the total variance, the sum of the columns' variances.
    ``pc<j>_loading_<feature>``
        Its unit loading on each column.
    ``pc<j>_score``
        The row's coordinate along it, ``e_j . (x - m)``, on the mean and
        the component before the row.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row.
    ``withheld_reason``
        Why a component was withheld, null where none was:
        ``"not_separated"`` where the data does not yet tell a component
        apart from a neighbor, ``"below_min_weight"`` under
        ``min_weight``.

    A component's four fields are null together while it is withheld.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`), ``TypeError`` for
    ``targets``, which this model does not take, and ``ValueError`` naming the
    problem for:

    - an ``n_components`` of 0 or past the features;
    - a ``method`` other than the three;
    - a ``learning_rate`` not finite or not above 0, or one given to a
      method other than ``"oja"``;
    - a ``max_state_bytes`` of 0 or above 64 MiB, or one the state's size
      passes;
    - a switch that reads residuals, such as ``emit_sigma``, and
      ``coef_every``.

    .. rubric:: The update

    Per row, ``w`` is the row's weight, ``λ = 0.5 ** (Δclock / half_life)``
    its decay (``lam ** Δclock`` under ``lam``), and ``x`` the row's
    features. Every method keeps the weight ``W``, the sum of the squared
    weights ``Q``, the mean ``m`` and the trace ``tr`` of the covariance
    ``C`` the components are those of, never ``C`` itself:

    .. code-block:: text

        W'   = λ * W + w           Q' = λ**2 * Q + w**2
        a    = λ * W / W'          b  = w / W'
        d    = x - m               m' = m + b * d
        C'   = a * C + a * b * d d'            (implied, not kept)
        tr'  = a * tr + a * b * |d|**2

    ``"svd"`` keeps ``q`` directions ``V`` (unit rows) with their singular
    values ``s``, so that ``C ≈ V' diag(s**2) V``. Each row stacks

    .. code-block:: text

        S = [ sqrt(a) * diag(s) V ]      q rows
            [ sqrt(a * b) * d'    ]      1 row

    so that ``S'S`` is the update above, and rotates ``S``'s rows in pairs
    until they are orthogonal (one-sided Jacobi): each row is then a singular
    value times a right singular vector, and the ``q`` longest are kept.
    This is the update scikit-learn's ``IncrementalPCA`` makes, one row a
    batch and in the mean form, with the decay and the weights.

    ``"ccipca"`` keeps each direction as ``v_i = l_i e_i``, its variance
    times its unit vector, and deflates the row as it goes (Weng, Zhang and
    Hwang, 2003):

    .. code-block:: text

        u_0     = d
        v_i'    = a * v_i + a * b * (u_i . e_i) * u_i
        u_{i+1} = u_i - (u_i . e_i') e_i'

    ``"oja"`` moves each direction toward the row and keeps them orthonormal
    in order by Gram-Schmidt (Oja, 1982), at a rate that is the row's share
    of the weight over the total variance, times the gain:

    .. code-block:: text

        eta  = learning_rate * b / tr'
        V'   = GramSchmidt(V + eta * (V d) d')
        l_i' = a * l_i + a * b * (e_i . d)**2

    A direction not yet started takes what is left of the first row the
    directions before it do not explain.

    .. rubric:: Example

    This code plants a stream of five columns whose variance lies mostly
    along one direction, and prints the leading component's share of the
    total variance before the last row:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(3)
        z = rng.standard_normal((20_000, 5)) * [3.0, 1.0, 0.5, 0.5, 0.5]
        x = z @ np.linalg.qr(rng.standard_normal((5, 5)))[0]
        cols = [f"x{i}" for i in range(5)]
        pl.DataFrame(x, schema=cols).write_parquet("five.parquet")

        pc = rp.spec.pca("pc", features=cols, half_life=float("inf"), n_components=2)
        out = pl.scan_parquet("five.parquet").rheon.fit_predict([pc]).collect().unnest("pc")
        last = out.row(-1, named=True)
        print("planted share: 0.84")
        print(f"found share: {last['pc0_share']:.2f}")

    .. code-block:: text

        planted share: 0.84
        found share: 0.84

    .. rubric:: See also

    - :func:`ew_cov`, whose ``pca`` gives the exact components of the
      ``k x k`` covariance, under a hard window too.

    .. rubric:: Notes

    **Which components are written.** Two components whose eigenvalues are
    equal are not identified: any rotation of the two fits the data as
    well. Component ``j`` is written while the data tells it apart from the
    component above it and the one below, by the likelihood-ratio test of
    the equality of two eigenvalues of a normal covariance (Anderson, 1963)
    at the 1% level: ``(n - 1) * ln(((l + m) / 2)**2 / (l * m))`` above
    9.21, the 99th percentile of a chi-squared with two degrees of freedom,
    where ``l`` and ``m`` are the two variances and ``n`` is Kish's
    effective size ``W**2 / Q``. A neighbor of no variance is told apart
    once ``n`` passes the rows that could leave it at none. So no count of
    rows gates the first rows: they are withheld while ``n`` is too small
    for the test, and a stream with tied components is withheld at the tie
    for as long as it lasts. Measured on a three-column stream with
    variances 100, 1 and 0.5 and no decay, the leading component was first
    written on the fifth row. On an isotropic stream of 5,000 rows, no row
    told its tied components apart.

    **Order and sign.** ``"svd"`` writes its components by variance, largest
    first. ``"ccipca"`` and ``"oja"`` write them in their deflation order,
    which is that order once they have converged. A component's sign is
    arbitrary, so the first time a direction is seen its largest loading in
    magnitude is positive, and on every row after it stays within 90° of
    the row before, so a score never flips sign from one row to the next.
    Compare a component with another library's up to sign.

    **Exactness.** ``"svd"`` with ``n_components`` equal to the number of
    features keeps every direction, so its components are the
    eigendecomposition of the decayed, weighted covariance itself, to
    rounding. With fewer it is exact while the data's rank is at most the
    directions kept, and otherwise drops, on every row, the variance outside
    them, which it never gets back: an approximation whose error grows as
    the gap below the kept directions closes. ``"ccipca"`` and ``"oja"``
    converge toward the components at a rate set by the gaps between them;
    without decay their error falls as rows arrive, and under decay it
    settles at a level the half-life sets.

    **No hard window.** None of the three methods can remove a row it has
    learned, so there is no ``window_size``. :func:`ew_cov`'s ``pca`` under a
    ``window_size`` gives the exact components of a window.

    **Cost.** A row costs ``O(q * k)`` under ``"ccipca"``, ``O(q**2 * k)``
    under ``"oja"``, and a few sweeps of ``O(q**2 * k)`` under ``"svd"``, for
    ``k`` features and ``q`` kept directions.
    """
    model: dict[str, Any] = {"type": "pca", "method": method}
    # Written only when given, as the Rust spec leaves each out when absent.
    if n_components is not None:
        model["n_components"] = n_components
    if learning_rate is not None:
        model["learning_rate"] = learning_rate
    if max_state_bytes is not None:
        model["max_state_bytes"] = max_state_bytes
    targets = _mirror_target(name, "pca", features, common, "its components are over")
    return _common(name, model, targets=targets, features=features, **common)
