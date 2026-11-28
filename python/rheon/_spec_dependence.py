"""The nonlinear dependence builder, :func:`dependence`: a part of
:mod:`rheon.spec`, split from ``_spec.py`` to keep it under the repository's
file-size cap. Import it from :mod:`rheon.spec`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Unpack

from rheon._duration import Duration
from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common, _listed, _who
from rheon._spec_words import quoted, refusal


def _target_names(targets: TargetList) -> list[str]:
    names: list[str] = []
    for t in targets:
        if isinstance(t, str):
            names.append(t)
        elif isinstance(t, Mapping):
            name = t.get("name") or t.get("column")
            if isinstance(name, str):
                names.append(name)
    return names


def _resolve_edges(
    who: str,
    param: str,
    role_singular: str,
    role_plural: str,
    names: Sequence[str],
    edges: dict[str, list[float]] | list[list[float]] | None,
) -> list[list[float]] | None:
    if edges is None:
        return None
    if isinstance(edges, dict):
        problems: list[Exception] = []
        missing = [c for c in names if c not in edges]
        if missing:
            problems.append(
                ValueError(
                    f"{who}: {param} must give edges for every {role_singular}, but it has none "
                    f"for {_listed([quoted(c) for c in missing])}. Add a list of edges for each, "
                    f"or pass a list of lists in the order of {role_plural}."
                )
            )
        extra = [c for c in edges if c not in names]
        if extra:
            verb = f"is not a {role_singular}" if len(extra) == 1 else f"are not {role_plural}"
            pronoun = "it" if len(extra) == 1 else "them"
            problems.append(
                ValueError(
                    f"{who}: {param} must name only {role_plural}, but it names "
                    f"{_listed([quoted(c) for c in extra])}, which {verb}. "
                    f"Remove {pronoun} from {param}, or add {pronoun} to {role_plural}."
                )
            )
        if problems:
            raise refusal(who, problems)
        return [list(edges[c]) for c in names]
    resolved = [list(e) for e in edges]
    if len(resolved) != len(names):
        raise ValueError(
            f"{who}: {param} must hold one list of edges for each {role_singular}, "
            f"{len(names)} in all, but it holds {len(resolved)}. Give one list for each "
            f"{role_singular}, in the order of {role_plural}."
        )
    return resolved


@_checked
def dependence(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    measures: list[Literal["mi", "mi_mm", "spearman", "kendall", "xi"]] | None = None,
    bins: int | None = None,
    target_bins: int | None = None,
    bin_rule: Literal["quantile", "fixed"] | None = None,
    bin_warm_rows: int | None = None,
    bin_warm_span: float | Duration | None = None,
    bin_edges: dict[str, list[float]] | list[list[float]] | None = None,
    target_bin_edges: dict[str, list[float]] | list[list[float]] | None = None,
    window_rows: int | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    rank_every: float | Duration | None = None,
    rank_every_rows: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Online nonlinear dependence between every feature and target pair.

    Use it to detect nonlinear, non-monotone, or rank relationships that linear
    correlation misses: parabolic, periodic, threshold, and heavy-tailed pairs.
    Every measure is read **before** the current row is learned, both in
    per-row struct fields and on demand as a tidy table via
    :meth:`rheon.polars.ModelBank.dependence`.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``, ``group`` and
    ``weight``, are in :mod:`rheon.spec`. ``half_life`` or ``lam`` decays the
    mutual-information tables; unset, every row counts alike (or inside the
    window, when ``window_rows`` or ``window_size`` is set). Finite decay is
    mutually exclusive with windows, with ``"mi_mm"``, and with rank measures.
    A row of weight 0 counts in no clock, no window, and no warm-up; a row
    whose target is null adds nothing to that target's pairs.

    This model's own parameters:

    ``features``
        Columns paired with every column of ``targets``.
    ``measures``
        Non-empty subset of ``"mi"`` (plug-in binned mutual information in
        nats), ``"mi_mm"`` (Miller-Madow bias-corrected MI, unit weights and
        no decay), ``"spearman"`` (windowed Spearman rank correlation),
        ``"kendall"`` (windowed Kendall's tau-b), and ``"xi"`` (windowed
        Chatterjee's xi). Unset defaults to ``["mi"]``.
    ``bins``, ``target_bins``
        Number of bins for each feature and each target when ``"mi"`` or
        ``"mi_mm"`` is active (from 2 to 256). Unset, ``bins`` defaults to 16
        and ``target_bins`` defaults to ``bins``. Refused beside ``bin_edges``
        or ``target_bin_edges``, or when neither MI measure is active.
    ``bin_rule``
        How interior bin edges are learned from the warm-up sample:
        ``"quantile"`` (the default, equal-weight quantiles) or ``"fixed"``
        (equal-width splits across the warm-up range).
    ``bin_warm_rows``
        Positive-weight rows held before bin edges are fixed and replayed into
        the joint tables (at least 2). Unset, defaults to
        ``max(ceil(62.5 * max(B_x, B_y)), 5 * B_x * B_y)`` (shrunk to
        ``window_rows`` under ``window_rows`` and capped to fit
        ``max_state_bytes``). Mutually exclusive with ``bin_warm_span``.
    ``bin_warm_span``
        Clock span held before bin edges are fixed (at least 2 positive-weight
        rows): a number of the clock column's units (rows when the spec has no
        clock), or a duration on a temporal clock. Checked live against
        ``max_state_bytes``. Mutually exclusive with ``bin_warm_rows``.
    ``bin_edges``, ``target_bin_edges``
        Fixed strictly increasing interior bin edges per feature (or target),
        as a dict keyed by column name or a list of lists in ``features`` (or
        ``targets``) order. When both are given, no warm-up is held and MI is
        ready from the first row.
    ``window_rows``
        Exact rolling window of the last this many positive-weight rows
        (at least 2). Required when any rank measure is active unless
        ``window_size`` is set. Refused beside ``window_size`` or finite decay.
    ``window_size``
        Exact rolling window of the last this long on the clock: a number of
        the clock column's units (rows when the spec has no clock), or a
        duration on a temporal clock. The data decides how many rows fall in
        the span, so state bytes are checked live against ``max_state_bytes``.
        Refused beside ``window_rows`` or finite decay.
    ``closed``
        Which edge of ``window_size`` keeps a row exactly that old, in Polars'
        words (``rolling_sum_by(closed=)``): ``"right"`` (the default, strictly
        less than ``window_size`` old) or ``"both"`` (less than or equal).
        Refused without ``window_size``.
    ``rank_every``, ``rank_every_rows``
        Refresh cadence for ``"spearman"`` and ``"xi"`` in per-row output:
        ``rank_every_rows`` refreshes every this many learned rows (default 1),
        and ``rank_every`` refreshes every this long on the clock. Held between
        refreshes; :meth:`rheon.polars.ModelBank.dependence` always computes
        fresh exact ranks from the current window. Mutually exclusive, and
        refused when neither ``"spearman"`` nor ``"xi"`` is in ``measures``.
    ``max_state_bytes``
        Maximum state bytes per group (default 256 KB, ``262144``; ceiling
        64 MiB). Fixed state sizes are checked when the spec is built; dynamic
        sizes (``window_size`` or ``bin_warm_span``) are checked as rows arrive,
        raising a notice once past half and stopping before exceeding the cap.

    .. rubric:: Output

    One output column, named after the spec, holds a record of these fields in
    each row. `docs/OUTPUTS.md#dependence
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#dependence>`_
    gives each field's type and when it is null:

    ``<measure>_<feature>_<target>``
        For each active measure in ``measures`` order, each feature, and each
        target: ``mi_<f>_<t>``, ``mi_mm_<f>_<t>``, ``spearman_<f>_<t>``,
        ``kendall_<f>_<t>``, ``xi_<f>_<t>``. Null while MI edges are still
        warming up, before enough observations exist, when a column is constant
        in the window, or when ``mi_mm`` sees any non-unit positive weight.
    ``weight_sum``
        Accumulated positive weight across learned rows (decayed, or inside the
        window) before the row.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`); ``withheld_reason`` is
        ``"below_warm_rows"`` while MI bin edges are still warming up.

    The same pairs can be read at any time as a tidy table with
    :meth:`rheon.polars.ModelBank.dependence`.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); and ``ValueError`` naming the
    problem for empty or unknown ``measures``, duplicate measures, rank
    measures without ``window_rows`` or ``window_size``, finite decay beside a
    window or ``"mi_mm"`` or ranks, both ``window_rows`` and ``window_size``,
    ``closed="both"`` without ``window_size``, invalid bin counts or non-monotone
    edges, unused binning or rank-cadence settings, or a fixed state exceeding
    ``max_state_bytes``. ``fit_predict`` raises if dynamic state under
    ``window_size`` or ``bin_warm_span`` would pass ``max_state_bytes``.

    .. rubric:: The update

    For each pair ``(x, y)`` over the active sample (all history, exponentially
    decayed, or the rolling window):

    .. code-block:: text

        MI     = sum_{n_ab > 0} (n_ab / N) * ln((n_ab / n_a) * (N / n_b))
        MI_MM  = MI - (B_xy - B_x - B_y + 1) / (2 * N)        [unit weights only]
        tau_b  = (n_c - n_d) / sqrt((n_0 - n_1) * (n_0 - n_2))
        rho    = corr(avg_rank(x), avg_rank(y))
        xi     = 1 - n * sum |r_{i+1} - r_i| / (2 * sum l_i * (n - l_i))

    ``MI`` is computed in nats from the ratio form with ``libm::log``, clamped
    at 0 against rounding noise; ``MI_MM`` subtracts the Miller-Madow
    finite-sample bias correction (matching ``sklearn.metrics.mutual_info_score``
    on the same discretized bins). ``tau_b`` is maintained incrementally in
    ``O(W)`` per row per pair; ``rho`` and ``xi`` match ``scipy.stats.spearmanr``
    and ``scipy.stats.chatterjeexi`` (with ties in ``x`` broken deterministically
    by arrival order, oldest first).

    .. rubric:: Example

    This code feeds a noisy parabola ``y = x**2`` where linear correlation is
    near zero, and reads nonlinear dependence from both the stream and the
    bank table:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(7)
        x = rng.uniform(-2.0, 2.0, 600)
        y = x**2 + 0.1 * rng.standard_normal(600)
        frame = pl.DataFrame({"x": x, "y": y})

        spec = rp.spec.dependence(
            "dep",
            targets=["y"],
            features=["x"],
            measures=["mi", "mi_mm", "spearman", "kendall", "xi"],
            bins=10,
            bin_warm_rows=200,
            window_rows=300,
        )
        bank = rp.ModelBank([spec])
        out = bank.fit_predict(frame)
        row = bank.dependence("dep").row(0, named=True)
        print(f"mi={row['mi']:.2f}, xi={row['xi']:.2f}, spearman={row['spearman']:+.2f}")

    .. code-block:: text

        mi=1.34, xi=0.87, spearman=+0.04

    .. rubric:: See also

    - :meth:`rheon.polars.ModelBank.dependence`, the tidy table of every pair's
      current nonlinear dependence readings.
    - :func:`marginal`, for linear correlation, slope, lags, and binned mean
      response curves across many features.
    - :func:`ew_cov`, for joint exponentially weighted covariance and linear
      correlation.
    """
    who = _who(name)
    feat_list = list(features)
    resolved_x_edges = _resolve_edges(who, "bin_edges", "feature", "features", feat_list, bin_edges)
    resolved_y_edges = _resolve_edges(
        who,
        "target_bin_edges",
        "target",
        "targets",
        _target_names(targets),
        target_bin_edges,
    )
    model: dict[str, Any] = {
        "type": "dependence",
        "measures": None if measures is None else list(measures),
        "bins": bins,
        "target_bins": target_bins,
        "bin_rule": bin_rule,
        "bin_warm_rows": bin_warm_rows,
        "bin_warm_span": bin_warm_span,
        "bin_edges": resolved_x_edges,
        "target_bin_edges": resolved_y_edges,
        "window_rows": window_rows,
        "window_size": window_size,
        "closed": None if closed == "right" else closed,
        "rank_every": rank_every,
        "rank_every_rows": rank_every_rows,
    }
    model = {k: v for k, v in model.items() if v is not None}
    return _common(name, model, targets=targets, features=feat_list, **common)
