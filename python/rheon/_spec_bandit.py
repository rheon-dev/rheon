"""The contextual bandit builder, :func:`bandit`: a part of
:mod:`rheon._spec` in a file of its own, keeping that one under
``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`rheon.spec` documents it with the rest.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._kwargs import CommonKwargs
from rheon._spec import _checked, _common
from rheon._spec_words import _SKIPPED_WHEN_ABSENT, quoted


@_checked
def bandit(
    name: str,
    *,
    features: list[str],
    arm: str,
    arms: list[str],
    reward: str,
    policy: Literal["linucb", "lints", "eps_greedy"] = "linucb",
    alpha: float | None = None,
    epsilon: float | None = None,
    l2: float = 1.0,
    seed: int = 0,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Disjoint linear contextual bandit over a logged arm and reward stream.

    Use it to recommend an arm from context features and evaluate linear
    bandit policies online over logged bandit feedback: LinUCB
    (``policy="linucb"``), linear Thompson sampling (``policy="lints"``), or
    linear epsilon-greedy (``policy="eps_greedy"``). Each arm maintains its
    own exponentially weighted ridge regression in square-root QR form, so
    every row's recommendation is read strictly before the row is learned.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``lam``, ``gap_cap``, ``min_weight``, ``embargo`` and
    ``group``, are in :mod:`rheon.spec`. ``reward`` and ``arm`` take the
    place of ``targets``. No diagnostic that reads regression residuals
    applies, and each such switch is refused by name. This model's own:

    ``features``
        Context feature columns read on every row.
    ``arm``
        The column that holds each row's logged arm, read as text (any dtype
        with a string form). On a row whose ``arm`` or ``reward`` is null,
        the model recommends an arm and scores every arm without updating any
        arm's fit.
    ``arms``
        Every arm name the bandit chooses among, in the order the
        ``score_<arm>`` fields and per-arm coefficients are written.
        Required, at least two, each unique and non-empty. A non-null logged
        arm not listed in ``arms`` is refused, naming the row.
    ``reward``
        The numeric column that holds each row's logged reward for the logged
        ``arm``.
    ``policy``
        The arm-selection rule:

        .. list-table::
           :header-rows: 1
           :widths: 22 78

           * - ``policy``
             - selection rule
           * - ``"linucb"`` (default)
             - upper confidence bound ``x' theta_a + alpha * ||R_a^-T x||_2``;
               deterministic given the state
           * - ``"lints"``
             - linear Thompson sampling drawing ``theta_tilde_a ~ N(theta_a,
               alpha^2 (R_a' R_a)^-1)`` from the counter-based Philox draw
               stream and choosing ``argmax_a x' theta_tilde_a``
           * - ``"eps_greedy"``
             - chooses a uniformly random arm with probability ``epsilon``
               and the greedy arm ``argmax_a x' theta_a`` with probability
               ``1 - epsilon``; emits the exact selection probability in
               ``prob``
    ``alpha``
        Exploration multiplier (at least 0) for ``"linucb"`` and ``"lints"``.
        Default ``1.0`` for those policies; ``0.0`` recovers pure greedy
        scoring. Refused under ``policy="eps_greedy"``.
    ``epsilon``
        Uniform exploration probability in ``[0, 1]`` for
        ``policy="eps_greedy"``. Default ``0.1``. Refused under
        ``policy="linucb"`` and ``policy="lints"``.
    ``l2``
        Per-arm ridge prior ``l2 > 0`` initializing ``R_a = sqrt(l2) * I``.
        Default ``1.0``.
    ``seed``
        Non-negative integer seed for the counter-based Philox4x64-10 draw
        stream used by ``"lints"`` and ``"eps_greedy"``. Default ``0``.
    ``max_state_bytes``
        Maximum serialized state size in bytes per group, at most 64 MiB.
        Default 256 KiB (``262_144``). Checked eagerly from the number of
        arms and features.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields, all but ``coef`` read before the row is learned.
    `docs/OUTPUTS.md#bandit
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#bandit>`_
    gives each field's type and when it is null:

    ``arm``
        The recommended arm, an ``Enum`` of ``arms`` in their order.
    ``score``
        The score of the recommended arm.
    ``score_<arm>``
        One per arm in ``arms`` order: the arm's UCB score (``"linucb"``),
        Thompson sample (``"lints"``), or greedy mean reward
        (``"eps_greedy"``).
    ``prob``
        Present only under ``policy="eps_greedy"``: the exact selection
        probability of the recommended arm (``1 - epsilon + epsilon / K`` for
        the greedy arm, ``epsilon / K`` for a non-greedy exploratory arm).
    ``weight_sum``
        The accumulated weight before the row and before its decay.
    ``settled_frac``
        How far the model had come toward its steady state under decay before
        the row, null where nothing decays.
    ``withheld_reason``
        Why the row's fields were withheld, null where they were not.
    ``coef``
        Per-arm linear coefficients after the row is learned, flattened in
        ``arms`` order, each holding ``[intercept] + features`` (or
        ``features`` when ``fit_intercept=False``). :func:`coef_index` maps
        each position to ``(arm, term)``.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`), ``TypeError`` for ``targets``,
    since this model takes ``reward`` and ``arm``, and ``ValueError`` naming
    the problem for:

    - fewer than two ``arms``, a duplicate arm, or an empty arm name;
    - a ``policy`` other than ``"linucb"``, ``"lints"`` or ``"eps_greedy"``;
    - ``alpha`` given with ``policy="eps_greedy"``, or ``epsilon`` given with
      ``policy="linucb"`` or ``"lints"``;
    - ``alpha < 0``, ``epsilon`` outside ``[0, 1]``, or ``l2 <= 0``;
    - ``reward`` or ``arm`` also listed in ``features``, or equal to each
      other;
    - a list-valued ``min_weight``, ``emit_target_weight=True``, or a
      residual-based switch such as ``emit_sigma``;
    - a configuration whose fixed state exceeds ``max_state_bytes``.

    .. rubric:: The update

    Each arm ``a`` maintains an upper-triangular square-root factor ``R_a``
    and transformed right-hand side ``u_a`` such that
    ``R_a' R_a = A_a`` and ``R_a' u_a = b_a``, initialized to
    ``R_a = sqrt(l2) I`` and ``u_a = 0``. With ``z = [1, x]`` (when
    ``fit_intercept=True``) and decay ``λ`` from ``half_life`` or ``lam``:

    .. code-block:: text

        theta_a = R_a^-1 u_a
        width_a = ||R_a^-T z||_2
        LinUCB:      score_a = z' theta_a + alpha * width_a
        LinTS:       score_a = z' (theta_a + alpha * R_a^-1 eps_a),   eps_a ~ N(0, I)
        eps_greedy:  score_a = z' theta_a

    On a row of weight ``w > 0`` with non-null ``(reward, arm) = (r, a*)``:

    .. code-block:: text

        R_a <- sqrt(λ) R_a,  u_a <- sqrt(λ) u_a         for every arm a
        [R_a*; u_a*] <- qr_givens([R_a*; sqrt(w) z'], [u_a*; sqrt(w) r])
        weight_sum <- λ weight_sum + w

    .. rubric:: Example

    .. code-block:: python

        import polars as pl
        import rheon.polars as rp

        df = pl.DataFrame({
            "x0": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            "arm": ["a", "b", "a", "b", "a", "b"],
            "reward": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        })
        spec = rp.spec.bandit(
            "b",
            features=["x0"],
            arm="arm",
            arms=["a", "b"],
            reward="reward",
            policy="linucb",
            half_life=50.0,
        )
        out = rp.ModelBank([spec]).fit_predict(df)
        print(out["b"].struct.field("arm").to_list()[:2])

    .. code-block:: text

        ['a', 'a']

    .. rubric:: See also

    - :func:`rls`, exponentially weighted recursive least squares for a
      numeric target without arm selection.
    - :func:`ew_class`, streaming Gaussian classification over labeled
      classes.

    .. rubric:: Notes

    **Deterministic draws.** Stochastic policies (``"lints"`` and
    ``"eps_greedy"``) draw from a counter-based Philox4x64-10 stream keyed by
    ``(seed, group)`` and counter-indexed by spec name and learned-row count.
    Zero-weight rows and :meth:`~rheon.polars.ModelBank.predict` calls do not
    advance the counter, so scoring and stepping match to the last bit and
    outputs are invariant to chunking, saving and restoring.
    """
    if "targets" in common:
        msg = (
            f"spec {quoted(name)}: bandit() takes no targets, since reward and arm name "
            "the reward and logged-arm columns. Pass those columns as reward and arm, and "
            "leave out targets."
        )
        raise TypeError(msg)
    model: dict[str, Any] = {
        "type": "bandit",
        "arms": arms,
        "policy": policy,
        "alpha": alpha,
        "epsilon": epsilon,
        "l2": l2,
        "seed": seed,
    }
    skipped = _SKIPPED_WHEN_ABSENT["bandit"]
    model = {k: v for k, v in model.items() if v is not None or k not in skipped}
    return _common(name, model, targets=[reward, arm], features=features, **common)
