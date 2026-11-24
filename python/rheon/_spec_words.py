"""The words a builder's parameter takes, read off its ``Literal``
annotation, and the one refusal of a word outside them; the parameters
where infinity means something; and the model keys a builder writes only
when they are given: a part of
:mod:`rheon._spec` moved to a file of its own, keeping that one
under ``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
"""

from __future__ import annotations

import json
import types
import typing
from collections.abc import Mapping, Sequence
from typing import Any, Literal


def quoted(x: Any) -> str:
    """A name or a word as a message quotes it: in double quotes, as JSON
    writes it, with any letter outside ASCII kept as it is, as the Rust side
    (``{:?}``) keeps it."""
    return json.dumps(x, ensure_ascii=False)


def _listed(items: Sequence[str]) -> str:
    """``a``, ``a or b``, ``a, b or c``: how a refusal lists what a
    parameter takes."""
    *rest, last = items
    return f"{', '.join(rest)} or {last}" if rest else last


def _one_of(words: Sequence[str]) -> str:
    """``"a"``, ``"a" or "b"``, ``"a", "b" or "c"``: how every refusal
    names the words a parameter takes, here and on the Rust side
    (``rheon_polars::words_listed``)."""
    return _listed([quoted(w) for w in words])


def _bad_word(v: Any, hint: Any) -> tuple[str, tuple[str, ...]] | None:
    """The word ``v`` gives outside the words ``hint`` takes, and those
    words, where ``v`` has the shape the hint asks for but a word it does
    not take: a str for a ``Literal``, an entry of a list of them, or a key
    of a dict keyed by them. ``None`` where the shape itself is wrong, which
    is a ``TypeError``, not a word outside a list."""
    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        for a in typing.get_args(hint):
            if (found := _bad_word(v, a)) is not None:
                return found
        return None
    if origin is Literal:
        words = typing.get_args(hint)
        if isinstance(v, str) and v not in words:
            return v, words
        return None
    if origin in (list, Sequence) and isinstance(v, (list, tuple)):
        (inner,) = typing.get_args(hint)
        if typing.get_origin(inner) is Literal and all(isinstance(x, str) for x in v):
            words = typing.get_args(inner)
            return next(((x, words) for x in v if x not in words), None)
        return None
    if origin in (dict, Mapping) and isinstance(v, Mapping):
        key, _ = typing.get_args(hint)
        if typing.get_origin(key) is Literal and all(isinstance(k, str) for k in v):
            words = typing.get_args(key)
            return next(((k, words) for k in v if k not in words), None)
    return None


#: The words renamed, refused naming the new one: ``ew_cov``'s statistic,
#: and the words of ``init`` and ``covariance_type`` that scikit-learn's
#: spellings replaced, the twin of the Rust side's ``RENAMED_VALUES`` (whose
#: other entry is a model ``type``, which no builder takes).
_RENAMED_WORDS = {
    "lagcorr": "lag_corr",
    "kmeanspp": "k-means++",
    "shared": "tied",
    "diagonal": "diag",
}

#: The refusal of a word for a parameter that takes one word alone, where
#: its model says why, as the Rust side words it (``rheon_core::RcovCfg``).
_WORD_REASONS = {
    ("rcov", "kernel"): (
        'kernel must be "parzen", the one kernel this model takes, but it is {got}. '
        'Set kernel to "parzen".'
    ),
}


#: The words a parameter refuses for a reason of its own, said beside the
#: words it takes: a model's ``closed`` takes two of Polars' four, and the
#: Rust side says why the other two cannot be.
_CLOSED_OUT = (
    "which leaves out the current row: a hard window ends at the last row the model learned, "
    "so that row is always inside it"
)
_REFUSED_FOR_A_REASON = {("closed", "left"): _CLOSED_OUT, ("closed", "none"): _CLOSED_OUT}


def word_reason(who: str, key: str, got: str) -> str | None:
    """The refusal of ``got`` for ``key`` where it has a reason of its own,
    as the Rust side words it; ``None`` for any other word."""
    why = _REFUSED_FOR_A_REASON.get((key, got))
    if why is None:
        return None
    return (
        f'{who}: {key} must be "right" or "both", but it is {quoted(got)}, {why}. '
        "Use one of those words."
    )


def words_refusal(who: str, key: str, got: str, words: Sequence[str], hint: Any = None) -> str:
    """The one refusal of a word outside the words a parameter takes
    (task 244, P-4), in the shape every message takes:
    ``spec "m": <key> must be "a", "b" or "c", but it is "x". Use one of
    those words.``, the shape the Rust side gives a spec dict and a TOML
    file too (``rheon_polars::words_body``). A parameter that also takes
    something other than a word (``hint``, its annotation) names that
    first: ``an int or "auto"``."""
    others = [a for a in typing.get_args(hint) if a in (int, float)]
    kinds = [{int: "an int", float: "a number"}[a] for a in others]
    taken = _listed(kinds + [quoted(w) for w in words])
    # A word renamed before 1.0 is told its new name, as the Rust side
    # tells it (`rheon_polars::RENAMED_VALUES`).
    new = _RENAMED_WORDS.get(got)
    if new is not None and new in words:
        return (
            f"{who}: {key} must be {taken}, but it is {quoted(got)}, the old name of "
            f"{quoted(new)}. Use {quoted(new)}."
        )
    if kinds:
        remedy = "Use one of those."
    elif len(words) == 1:
        remedy = f"Use {quoted(words[0])}."
    else:
        remedy = "Use one of those words."
    return f"{who}: {key} must be {taken}, but it is {quoted(got)}. {remedy}"


def refusal(who: str, problems: Sequence[Exception]) -> Exception:
    """Every problem a builder's check found, as one exception to raise,
    in the shape the Rust side gives a spec's (``render_refusal``): one
    problem is its own message, and several are listed one to a line under
    ``spec "m": N problems.``, the spec named once. A ``TypeError`` if any
    problem is one, since a value of the wrong kind is the first to fix,
    and a ``ValueError`` otherwise."""
    prefix = f"{who}: "
    lines: list[str] = []
    for p in problems:
        line = str(p).removeprefix(prefix)
        if line not in lines:
            lines.append(line)
    if len(lines) == 1:
        return problems[0]
    kind = TypeError if any(isinstance(p, TypeError) for p in problems) else ValueError
    listed = "".join(f"\n- {line}" for line in lines)
    return kind(f"{prefix}{len(lines)} problems.{listed}")


# The parameters where ``inf`` means something: no decay, no ceiling, a
# pinned coefficient, no clip, least squares (``huber_delta``), a step nothing
# caps (``pa``'s ``c``), the argmin (``average_eta``). Rust parses these as
# ``Num`` and ``validate`` takes them at ``inf``; everywhere else ``validate``
# says ``finite``, and an infinity is refused here by name first, because
# once it is inside the model's tagged union serde can only say `expected
# f64`. The table was the fields whose Rust *type* admits ``inf``, which named
# ``ridge`` and ``q``, both refused by ``validate``, and missed six that mean
# something (review 2026-09-12, S27). Keyed by builder, since one name can be
# a limit for one builder and no setting for another; ``"*"`` is the shared
# parameters. tests/test_error_messages.py checks this table against the
# Rust side.
#: Where ``inf`` meant something until task 120 (2026-09-28), and the Rust
#: side now refuses it with a message that names what to write instead:
#: ``half_life = "inf"`` or a finite cap, and ``"reset"``. The builder passes
#: these to it rather than say only "must be finite". ``min_weight`` joined
#: them in task 278 (the user, 2026-10-25): a floor at ``inf`` withheld every
#: row for good, and the Rust side's refusal says why.
_INF_REFUSED_BY_RUST = frozenset({"gap_cap", "session_gap", "min_weight"})

#: ``holt``'s keys written only when given, as the Rust spec skips them
#: when absent: the trend switch and the seasonal forms' settings.
_HOLT_OPTIONAL = frozenset(
    {
        "trend",
        "horizons",
        "seasonal",
        "seasonal_periods",
        "season_length",
        "seasonal_half_life",
        "damping_half_life",
        "start",
        "initial_level",
        "initial_trend",
        "initial_seasonal",
        "max_state_bytes",
    }
)

_INF_OK: dict[str, frozenset[str]] = {
    "*": frozenset(
        {
            "half_life",
            "average_eta",
            # The noise gate at `inf` is off: no ratio is above it.
            "max_error_inflation",
            # A window budget at `inf` is no bound (`{"thin": inf}`).
            "window_budget",
            # A solve cadence at `inf` never comes due: the sums are
            # collected and nothing is solved (task 278).
            "solve_every",
            # A diagnostic's memory at `inf` is the run-once form (task 221).
            "calibration_half_life",
            "breaks_half_life",
            "robust_se_half_life",
            "specification_half_life",
            "tails_half_life",
            "influence_half_life",
            "feature_health_half_life",
        }
    ),
    "ewridge": frozenset({"long_half_life"}),
    "lasso": frozenset({"select_half_life"}),
    "kalman": frozenset({"coef_half_life", "revert_half_life"}),
    # A refit budget at `inf` is no bound (task 277).
    "huber": frozenset({"huber_delta", "huber_scale_half_life", "refit_budget"}),
    "sgd": frozenset(
        {
            "clip_gradient",
            "coef_min",
            "coef_max",
            "huber_delta",
            "huber_scale_half_life",
            "refit_budget",
        }
    ),
    "quantile": frozenset({"refit_budget"}),
    "pa": frozenset({"c", "coef_min", "coef_max"}),
    "holt": frozenset({"trend_half_life", "seasonal_half_life", "damping_half_life"}),
    # No cap on a row's step; no bound on the refit buffers.
    "glm": frozenset({"step_cap", "refit_budget"}),
    # No bound on the refit buffers.
    "expectile": frozenset({"refit_budget"}),
    # No bound on the bins' memory (docs/PLAN.md task 131).
    "marginal": frozenset({"bin_budget", "bin_huber_delta", "refit_budget"}),
    # A cadence on the clock at `inf` never comes due by the clock: its count
    # alone fires it (task 538).
    "kmeans": frozenset({"update_every", "split_merge_every"}),
    "corrchange": frozenset({"permute_every"}),
}


#: The model keys the Rust spec skips when they are absent
#: (``skip_serializing_if`` in ``crates/rheon-polars/src/spec.rs``): a key
#: written as null there would move every spec's bytes and cost a schema
#: bump. A builder writes each of these only when it is given, so the bank
#: reports back the dict it made: a ``None`` the bank never writes made
#: ``ModelBank([s]).specs[0] != s`` for every ``marginal`` (review
#: 2026-10-05, the TB5 leg). ``holt``'s ``trend`` is written only when off,
#: and its seasonal forms' settings only when given.
_SKIPPED_WHEN_ABSENT: dict[str, frozenset[str]] = {
    "holt": _HOLT_OPTIONAL,
    "hawkes": frozenset(
        {"kernel_half_lives", "window_events", "window_size", "closed", "max_state_bytes"}
    ),
    "garch": frozenset(
        {
            "vol",
            "dist",
            "truncation",
            "variance_targeting",
            "start_rows",
            "start_span",
            "horizons",
            "max_horizon_work",
            "max_state_bytes",
        }
    ),
    "ewridge": frozenset({"gram_threads"}),
    "ar": frozenset({"ridge", "max_state_bytes", "select"}),
    "pca": frozenset({"n_components", "learning_rate", "max_state_bytes"}),
    "tdigest": frozenset({"compression", "max_state_bytes"}),
    "ddsketch": frozenset({"alpha", "window_rows", "window_size", "closed", "max_state_bytes"}),
    "isotonic": frozenset(
        {
            "increasing",
            "y_min",
            "y_max",
            "out_of_bounds",
            "window_rows",
            "window_size",
            "closed",
            "max_state_bytes",
        }
    ),
    "expectile": frozenset(
        {
            "ridge",
            "refit_rows",
            "refit_span",
            "window_rows",
            "window_size",
            "closed",
            "max_state_bytes",
            "refit_budget",
        }
    ),
    "huber": frozenset(
        {
            "huber_scale",
            "huber_scale_half_life",
            "feature_sets",
            "ridge_scale",
            "emit_huber_scale",
            "huber_refit_rows",
            "huber_refit_span",
            "max_state_bytes",
            "refit_budget",
        }
    ),
    "sgd": frozenset(
        {
            "huber_scale",
            "huber_scale_half_life",
            "huber_refit_rows",
            "huber_refit_span",
            "max_state_bytes",
            "refit_budget",
        }
    ),
    "quantile": frozenset(
        {"quantile_refit_rows", "quantile_refit_span", "max_state_bytes", "refit_budget"}
    ),
    "ew_cov": frozenset({"gram_threads"}),
    "marginal": frozenset(
        {
            "lags",
            "serial_rule",
            "cross_lags",
            "bins",
            "bin_rule",
            "bin_warm_rows",
            "bin_edges",
            "bin_budget",
            "bin_stat",
            "bin_huber_delta",
            "bin_refit_rows",
            "refit_budget",
            "shards",
            "window_lags",
            "feature_moments",
        }
    ),
    "audit": frozenset({"pairs", "distinct_cap"}),
    "kalman": frozenset({"covariance_form", "horizons"}),
    "glm": frozenset({"refit_span", "max_state_bytes"}),
    "kmeans": frozenset({"warm_span", "update_every", "split_merge_every"}),
    "hmm": frozenset({"warm_span"}),
    "bocpd": frozenset({"warm_span", "hazard_rows"}),
    "corrchange": frozenset({"permute_every", "window_size", "monitor_span", "max_state_bytes"}),
    "realized": frozenset(
        {"window_rows", "window_size", "closed", "lm_window", "lm_beta", "max_state_bytes"}
    ),
    "power_variation": frozenset({"window_rows", "window_size", "closed", "max_state_bytes"}),
}
