"""The online kernel recursive least squares builder, :func:`krls`: a part of
:mod:`rheon.spec`, split from ``_spec.py`` to keep it under the repository's
file-size cap. Import it from :mod:`rheon.spec`.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._duration import Duration
from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common


@_checked
def krls(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    kernel: Literal["rbf", "laplacian", "poly", "linear"] = "rbf",
    gamma: float | None = None,
    degree: int | None = None,
    coef0: float | None = None,
    ridge: float = 1.0,
    max_atoms: int | None = None,
    ald_threshold: float | None = None,
    window_rows: int | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    emit_n_atoms: bool = False,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Online kernel ridge regression in a reproducing kernel Hilbert space,
    fitted one row at a time.

    Use it when the relationship between the features and the target is smooth
    and nonlinear -- interactions, curves, radial patterns -- and a linear
    model such as :func:`ewridge` or :func:`rls` leaves structure in the
    residuals. Each row is predicted from the state before it arrives, then
    learned, so the row's own target never enters its own prediction.

    Two forgetting modes are supported:

    - **ALD dictionary mode** (the default, when neither ``window_rows`` nor
      ``window_size`` is set): keeps a bounded dictionary of up to
      ``max_atoms`` representative rows admitted by Engel's approximate linear
      dependence (ALD) criterion, and solves kernel ridge regression in the
      whitened subspace spanned by those atoms, optionally with exponential
      forgetting (``half_life`` or ``lam``).
    - **Sliding-window mode** (``window_rows`` or ``window_size``): keeps
      every positive-weight learned row inside the window and solves exact
      kernel ridge regression on those rows via a Givens-updated Cholesky
      factor, matching scikit-learn's ``KernelRidge(alpha=ridge)`` on the
      window's rows.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``, ``group``,
    ``weight``, ``fit_intercept``, ``embargo``, ``label_clock`` and the
    residual diagnostics, are in :mod:`rheon.spec`. ``half_life`` or ``lam``
    defaults to ``inf`` (no decay) when omitted; finite decay is supported in
    ALD mode and refused beside ``window_rows`` or ``window_size``.
    ``fit_intercept`` defaults to ``True``: the intercept is unpenalized,
    centered over the weighted rows in ALD mode and eliminated exactly in
    sliding-window mode. ``ridge`` is on the sum scale (matching
    ``sklearn.kernel_ridge.KernelRidge(alpha=ridge)``), so scaling every row
    weight by ``c`` scales the data term relative to ``ridge`` just as in
    ``KernelRidge(sample_weight=...)``. A row of weight 0 counts in no clock,
    adds no dictionary atom and enters no window, while still being scored.

    This model's own parameters:

    ``features``
        One or more feature columns. Because radial and polynomial kernels
        depend on feature distances and inner products, put features on
        comparable scales beforehand if their units differ widely.
    ``kernel``
        Kernel family:

        - ``"rbf"`` (default): ``exp(-gamma * ||x - y||²)``
        - ``"laplacian"``: ``exp(-gamma * ||x - y||_1)``
        - ``"poly"``: ``(gamma * x'y + coef0) ** degree``
        - ``"linear"``: ``x'y``
    ``gamma``
        Inverse length scale for ``"rbf"`` and ``"laplacian"``, or inner
        product scale for ``"poly"``; finite and above 0. Unset, defaults to
        ``1 / len(features)``. Refused when ``kernel="linear"``.
    ``degree``
        Polynomial degree for ``kernel="poly"``; integer at least 1, default
        ``3``. Refused for any other kernel.
    ``coef0``
        Non-negative polynomial offset for ``kernel="poly"``; finite and at
        least 0, default ``1.0``. Refused for any other kernel.
    ``ridge``
        Positive sum-scale ridge penalty ``r > 0`` (finite, default ``1.0``).
        Matches ``alpha`` in ``sklearn.kernel_ridge.KernelRidge``.
    ``max_atoms``
        Maximum number of dictionary atoms kept in ALD mode (integer at least
        1, default ``100``). Checked against ``max_state_bytes`` when the spec
        is built. Refused beside ``window_rows`` or ``window_size``.
    ``ald_threshold``
        Relative approximate linear dependence threshold in ``[0, 1)``
        (default ``1e-3``). A new row is admitted to the dictionary when the
        squared projection residual in kernel space exceeds
        ``ald_threshold * k(x, x)`` and the dictionary has fewer than
        ``max_atoms`` atoms. Refused beside ``window_rows`` or
        ``window_size``.
    ``window_rows``
        Sliding window of the last this many positive-weight rows (integer at
        least 1). Checked against ``max_state_bytes`` when the spec is built.
        Refused beside ``window_size``, ``max_atoms``, ``ald_threshold``, or
        finite ``half_life`` / ``lam < 1``.
    ``window_size``
        Sliding window of the rows of the last this long on the clock: a
        finite positive number of the clock column's units (rows when the spec
        has no clock), or a duration on a temporal clock. Rows leave when they
        reach ``window_size`` old on the decayed clock (exact stamps). Because
        the data decides how many rows fall inside the span, state bytes are
        checked as rows arrive (see ``max_state_bytes``). Refused beside
        ``window_rows``, ``max_atoms``, ``ald_threshold``, or finite
        ``half_life`` / ``lam < 1``.
    ``closed``
        Which edge of ``window_size`` holds a row exactly that old, in Polars'
        words (``rolling_sum_by(closed=)``). ``"right"`` (default) keeps rows
        strictly less than ``window_size`` old; ``"both"`` keeps rows at most
        ``window_size`` old and is refused without ``window_size``.
    ``emit_n_atoms``
        When ``True``, add ``n_atoms`` (``Int64``) to the output struct: the
        dictionary size (or window size) before the row is learned. Default
        ``False`` (field omitted).
    ``max_state_bytes``
        Maximum bytes a group's state may hold. Default 256 KB (``262144``);
        ceiling 64 MiB (``67108864``). Checked at build time for ALD mode and
        ``window_rows``; checked per row under ``window_size``, raising one
        notice past half the cap and stopping with an error before exceeding
        the cap.

    .. rubric:: Output

    One output column, named after the spec, holds a record of these fields
    in each row. `docs/OUTPUTS.md#krls
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#krls>`_
    gives each field's type and when it is null:

    ``pred_<t>``, ``resid_<t>``
        Per target: the kernel ridge prediction from the rows before this one,
        and ``y - pred_<t>`` where the target is present. Null before the
        first learned row, and under ``window_size`` when every learned row
        has aged out of the window.
    ``weight_sum``
        Accumulated weight before the row (decayed in ALD mode, or the sum of
        weights inside the window).
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`); ``withheld_reason``
        is ``"not_identified"`` before the first learned row or when a clock
        window is empty.
    ``n_atoms``
        Added when ``emit_n_atoms=True`` (``Int64``): number of dictionary
        atoms or window rows before the row is learned; null on skipped rows.

    Plus any residual diagnostics switched on (``emit_sigma``,
    ``emit_zscore``, ``emit_metrics``, ``conformal``, ``resid_quantiles``,
    ``emit_autocorr``, ``emit_drift``, ``emit_calibration``, ``emit_breaks``,
    ``emit_specification``, ``emit_tails``, ``emit_feature_health``,
    ``emit_clocks``, ``emit_target_weight``, and ``emit_selected`` /
    ``emit_averaged`` over a ``half_life`` grid). ``krls`` has no linear
    ``coef`` vector, so ``coef_every``, ``emit_se_coef``, ``emit_robust_se``,
    ``emit_influence``, ``emit_error_inflation`` and ``max_error_inflation``
    are refused.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); and ``ValueError`` naming the
    problem for an unknown ``kernel``, ``gamma`` / ``degree`` / ``coef0`` on
    a kernel that does not take them, non-positive or non-finite ``ridge`` or
    ``gamma``, ``degree < 1``, ``coef0 < 0``, ``max_atoms < 1``,
    ``ald_threshold`` outside ``[0, 1)``, ``window_rows < 1``, non-positive or
    non-finite ``window_size``, both windows together, ``closed="both"``
    without ``window_size``, ALD parameters or finite decay beside a window,
    ``max_state_bytes`` of 0 or above 64 MiB, or a fixed configuration whose
    state exceeds ``max_state_bytes``.

    .. rubric:: The update

    **ALD mode** maintains a dictionary ``D = {d_1, ..., d_m}`` of at most
    ``max_atoms`` rows and the lower Cholesky factor ``L_D L_D' = K_DD`` of
    their Gram matrix. Every row ``x`` is mapped to whitened Nystrom
    coordinates ``psi(x) = L_D^-1 k_D(x)``. The squared residual norm in the
    RKHS is ``delta = k(x, x) - ||psi(x)||²``; when ``m < max_atoms`` and
    ``delta > ald_threshold * k(x, x)`` (with ``delta > 0``), ``x`` is
    appended to ``D``, extending ``L_D`` by one row with diagonal
    ``sqrt(delta)`` and padding ``psi`` by one entry without dividing by
    ``delta``. The regularized normal equations in the whitened basis are:

    .. code-block:: text

        fit_intercept=False:  (S + r I) theta = b,    pred(x) = psi(x)' theta
        fit_intercept=True:   (S_c + r I) theta = b_c,
                              pred(x) = y_bar + (psi(x) - psi_bar)' theta

    where ``S = sum w_i psi_i psi_i'`` and ``b = sum w_i psi_i y_i`` (with
    weighted centering ``S_c = S - W psi_bar psi_bar'`` and
    ``b_c = b - W psi_bar y_bar``). Without decay, the upper Cholesky factor
    ``L_s`` of ``S_c + r I`` is updated in ``O(m²)`` per row by Givens
    rotations; under exponential decay ``S`` and ``b`` decay by ``d`` each
    step and refactor in ``O(m³/3)`` when needed.

    **Sliding-window mode** maintains the exact lower Cholesky factor
    ``L_W L_W' = W^{-1} (r I) + K_WW`` (scaled as ``L_W L_W' = r W^{-1} + K``
    in weighted form ``sqrt(W) K sqrt(W) + r I``) over the ``w`` rows
    currently inside the window. Dropping the oldest row performs a row/column
    deletion via orthogonal Givens rotations (an exact positive rank-1 update
    with no subtraction drift) in ``O(w²)``; appending a new row extends
    ``L_W`` by one row in ``O(w²)``. With ``fit_intercept=True``, the
    unpenalized intercept is eliminated via the exact Schur complement on
    the ones vector before solving for the dual weights.

    .. rubric:: Example

    This code fits a nonlinear sine-plus-quadratic surface online and compares
    its held-out predictions against the true values:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(0)
        n = 400
        x0 = rng.uniform(-2.0, 2.0, n)
        x1 = rng.uniform(-2.0, 2.0, n)
        y = np.sin(x0) + 0.5 * x1**2
        frame = pl.DataFrame({"x0": x0, "x1": x1, "y": y})

        model = rp.spec.krls("k", targets=["y"], features=["x0", "x1"], ridge=0.01)
        out = rp.ModelBank([model]).fit_predict(frame).unnest("k")
        err = (out["resid_y"].tail(100) ** 2).mean()
        print(round(float(err), 4))

    .. code-block:: text

        0.0001

    .. rubric:: See also

    - :func:`ewridge`, for linear ridge regression with full coefficient
      diagnostics and leverage gates.
    - :func:`isotonic`, for nonparametric monotone calibration on one feature.
    - :func:`marginal`, for binned univariate curves per feature.
    """
    model: dict[str, Any] = {
        "type": "krls",
        "kernel": None if kernel == "rbf" else kernel,
        "gamma": gamma,
        "degree": degree,
        "coef0": coef0,
        "ridge": None if ridge == 1.0 else ridge,
        "max_atoms": max_atoms,
        "ald_threshold": ald_threshold,
        "window_rows": window_rows,
        "window_size": window_size,
        "closed": None if closed == "right" else closed,
        "emit_n_atoms": True if emit_n_atoms else None,
    }
    # Each absent when at its default, as the Rust spec skips it, so a bank
    # reports back the dict this made (`ModelBank.specs`).
    model = {k: v for k, v in model.items() if v is not None}
    return _common(name, model, targets=targets, features=list(features), **common)
