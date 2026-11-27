"""The keyword parameters every spec builder shares, as one ``TypedDict``
(PEP 692).

The builders in :mod:`rheon.spec` take the shared parameters as
``**common``, annotated as ``Unpack[CommonKwargs]``. So an editor completes
them and a type checker finds a misspelled one, where ``**kwargs: Any``
would show nothing and leave a typo to be found when the call runs. A
``TypedDict`` has no defaults: each builder's signature gives them, and
:mod:`rheon.spec` describes each parameter.
"""

# Notes for whoever changes this file (docs/IMPROVEMENTS.md U4):
#
# `CommonKwargs` is pinned to the shared parameters by
# tests/test_kwargs_typing.py: same keys, same annotations, same required
# set. Change a shared parameter and the test says so.
#
# There were once 21 further classes here, one per model, mirroring each
# builder's *own* parameters to type the expression namespace's `**kwargs`.
# Task 85 removed that namespace, and the test that held each class to its
# builder went with it. What was left was 21 unconsumed copies of builder
# signatures with nothing holding them to the originals -- the drift the
# test above exists to prevent. They were deleted on 2026-09-17; rebuild
# them from the builders if a typed keyword surface is ever wanted again.
# The namespace's own set, every shared parameter but `group` and
# `group_close`, survived as a base class (`ExprKwargs`) until task 239
# folded it into `CommonKwargs`: nothing but `CommonKwargs` read it.
#
# No `from __future__ import annotations` here: under it `Required[...]` is
# a string the TypedDict machinery does not look inside, so every key would
# be optional at runtime and the test could not see the required ones.

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

from rheon._duration import Duration

__all__ = [
    "CommonKwargs",
]


class CommonKwargs(TypedDict, total=False):
    """What the builders take as ``**common``: every parameter the models
    share, the group and its close policy among them."""

    fit_intercept: bool
    clock: str | None
    half_life: float | Duration | list[float | Duration] | None
    lam: float | None
    gap_cap: float | Duration | None
    restart_after_step_back: float | Duration | None
    session: str | None
    session_gap: float | Duration | None
    weight: str | None
    min_weight: float | list[float] | None
    min_settled_frac: float | None
    max_error_inflation: float | None
    emit_error_inflation: bool
    emit_se_coef: bool
    emit_clocks: bool
    emit_target_weight: bool
    coef_every: float | Duration | None
    max_rows_between_coefs: int | None
    emit_sigma: bool
    emit_zscore: bool
    emit_selected: bool
    emit_averaged: bool
    average_eta: float | None
    emit_metrics: (
        bool
        | list[
            Literal[
                "ic",
                "r2",
                "hit_rate",
                "mae",
                "pinball",
                "log_loss",
                "coverage",
                "width",
                "winkler",
            ]
        ]
    )
    metric_quantiles: list[float] | None
    conformal: float | list[float] | None
    conformal_symmetric: bool
    conformal_rate: float | Literal["auto"] | None
    emit_conformal_p: bool
    conformal_p_window_rows: int | None
    conformal_p_window_size: float | Duration | None
    max_state_bytes: int | None
    resid_quantiles: list[float] | None
    emit_autocorr: bool
    resid_autocorr_lag: int | None
    emit_drift: bool
    drift_delta: float | None
    drift_threshold: float | Duration | None
    drift_action: Literal["flag", "reset"]
    drift_kind: Literal["page_hinkley", "cusum", "adwin", "ddm", "eddm", "kswin"] | None
    drift_options: Mapping[str, float | int | bool | str] | None
    emit_calibration: bool
    emit_spiegelhalter: bool
    calibration_half_life: float | Duration | None
    emit_breaks: bool
    breaks_half_life: float | Duration | None
    emit_robust_se: bool
    robust_se_half_life: float | Duration | None
    robust_se_lags: int | None
    robust_se_types: list[Literal["hc0", "hc1", "hc2", "hc3"]] | None
    emit_specification: bool
    emit_arch_lm: bool
    arch_lm_lags: int | None
    emit_white: bool
    white_interaction_terms: bool | None
    specification_half_life: float | Duration | None
    ljung_box_lags: int | None
    horizon_rows: int | None
    emit_tails: bool
    tails_half_life: float | Duration | None
    emit_influence: bool
    emit_cooks_distance: bool
    emit_dfbetas: bool
    influence_half_life: float | Duration | None
    emit_feature_health: bool
    feature_health_half_life: float | Duration | None
    embargo: float | Duration | None
    label_clock: str | Sequence[str] | None
    max_held_rows: int | None
    held_overflow: Literal["refuse", "drop_oldest"] | None
    group: str | None
    group_close: Literal["monotone", "session"] | None
