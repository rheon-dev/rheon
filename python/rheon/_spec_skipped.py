"""The model keys a builder writes only when given: a part of
:mod:`rheon.spec`'s plumbing, split from ``_spec.py`` to keep it under the
repository's file-size cap.
"""

from __future__ import annotations

#: The model keys the Rust spec skips when they are absent
#: (``skip_serializing_if`` in ``crates/rheon-polars/src/spec.rs``): a key
#: written as null there would move every spec's bytes and cost a schema
#: bump. A builder writes each of these only when it is given, so the bank
#: reports back the dict it made: a ``None`` the bank never writes made
#: ``ModelBank([s]).specs[0] != s`` for every ``marginal`` (review
#: 2026-10-05, the TB5 leg). ``holt``'s ``trend`` is written only when off.
_SKIPPED_WHEN_ABSENT: dict[str, frozenset[str]] = {
    "holt": frozenset({"trend"}),
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
            "horizon",
            "max_state_bytes",
        }
    ),
    "tdigest": frozenset({"compression", "max_state_bytes"}),
    "ewridge": frozenset({"gram_threads"}),
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
    "kalman": frozenset({"covariance_form"}),
    "glm": frozenset({"refit_span", "max_state_bytes"}),
    "kmeans": frozenset({"warm_span", "update_every", "split_merge_every"}),
    "hmm": frozenset({"warm_span"}),
    "bocpd": frozenset({"warm_span"}),
    "corrchange": frozenset({"permute_every"}),
}
