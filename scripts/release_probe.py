"""The workload `scripts/compare_release.py` runs under two builds.

Self-contained, and written against the API every release since 0.8 has:
numeric clocks, and every builder called by keyword. Each spec is built and
run on its own, so a spec a build refuses -- a model or a parameter it does
not have -- is recorded as refused rather than ending the run. Writes every
output field of every spec, and the frame `closed_groups` drains, to a
parquet file, and what ran, each spec's kind and the count of each hard
path the run reached (`PATHS`) to a JSON manifest. Every spec of
`WORKLOAD` and `HARD_PATHS` runs four times: on `stream()`; on it with
each group's first row at weight 0 (`w0_`), and with it null (`null0_`);
and saved at `RESUME_AT`, loaded and continued (`resume_`):

    python scripts/release_probe.py OUT.parquet MANIFEST.json

With ``--states``, it fits each spec on the first half of the stream and
saves the bank under the spec's name instead, for
``tests/test_released_state.py``: a released build writes the files, and
this build loads them and goes on:

    python scripts/release_probe.py --states DIR MANIFEST.json

With ``--resume``, a build loads the files ``--states`` wrote, each with
its spec as that build makes it, fits the second half of the stream and
writes what it reports there, and the tables ``tables`` reads, to
``<OUT>/<name>.<table>.parquet``: a released build's own continuation of its
own files, which ``tests/test_released_state.py`` holds this build's to:

    python scripts/release_probe.py --resume DIR OUT NAME...
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import warnings

import numpy as np
import polars as pl

# This build is `rheon.polars`; a release before 2.0 is `polars_online`. Chosen
# by which is installed, so an import error inside `rheon` is shown, not hidden
# behind a missing `polars_online`.
po = importlib.import_module(
    "rheon.polars" if importlib.util.find_spec("rheon") is not None else "polars_online"
)

INF = float("inf")


def stream(n: int = 400, seed: int = 7) -> pl.DataFrame:
    """Three interleaved groups on an irregular clock with gaps past the
    cap, a session change halfway, null features (skipped rows), null
    targets (predict-only rows), zero weights, a 0/1 label, a class label,
    and a block key that only increases, for the models that close groups.

    Two columns serve `HARD_PATHS` alone, derived from the others so no
    other column's draws move: `x1b`, `x1` stepped up by 4 from row 220, a
    break for `bocpd`'s fold to follow; and `ts_decade`, `ts` with each
    group's first row of positive weight a decade earlier, so each group's
    temporal clock runs at a decade's age from its next row on. The rows of
    weight 0 are all in the first half."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.choice([0.5, 1.0, 1.5, 2.0], size=n))
    # Two gaps past the cap. Raising rows 90 and 250 alone, as this did until
    # task 120, stepped the clock back on the row after each, which the
    # removed `"max"` absorbed and the default now refuses.
    t[90:] += 40.0
    t[250:] += 40.0
    x = rng.standard_normal((n, 3))
    y = 0.5 + x @ np.array([1.0, -0.5, 0.0]) + 0.3 * rng.standard_normal(n)
    z = 0.2 * y + x[:, 2] + 0.3 * rng.standard_normal(n)
    w = rng.uniform(0.5, 1.5, n)
    w[rng.choice(n, 15, replace=False)] = 0.0
    # Rows of weight 0 in the first half alone: a row of weight 0 counts in
    # no clock since 539 (hard rule 9), so a release's own continuation of
    # its saved half, over rows that all count, is the one this build's must
    # meet, the state the release saved being the only difference.
    w[HALF:][w[HALF:] == 0.0] = 1.0
    x0 = x[:, 0].copy()
    x0[[30, 31, 200]] = np.nan
    yy = y.copy()
    yy[[45, 46, 300]] = np.nan
    # A clock that steps back by 30 at row 150: a restart under
    # `restart_after_step_back = 10`.
    t_back = t.copy()
    t_back[150:] -= 30.0
    return (
        pl.DataFrame(
            {
                "t": t,
                "g": np.array(["a", "b", "c"])[np.arange(n) % 3],
                "s": np.where(np.arange(n) < n // 2, "am", "pm"),
                "block": np.arange(n) // 50,
                "x0": x0,
                "x1": x[:, 1],
                "x2": x[:, 2],
                "y": yy,
                "z": z,
                "w": w,
                "hit": (y > 0.5).astype(float),
                "lab": np.where(y > 0.5, "up", "down"),
                "t_back": t_back,
            }
        )
        .with_columns(
            pl.col("x0", "y").fill_nan(None),
            # `t` as a time of day: the temporal clock and its duration
            # parameters, on the same steps.
            ts=pl.datetime(2024, 1, 2, 9, 30) + pl.duration(milliseconds=(pl.col("t") * 1000)),
            x1b=pl.col("x1") + pl.when(pl.int_range(pl.len()) >= BREAK_AT).then(4.0).otherwise(0.0),
            # A positive amount for `glm`'s gamma and Tweedie families.
            amount=(0.3 * pl.col("x1") - 0.2 * pl.col("x2")).exp(),
            **{f"r{c}": pl.Series(v) for c, v in zip("abc", _returns(n).T, strict=True)},
            **{f"v{j}": pl.Series(v) for j, v in enumerate(_wide(n).T)},
        )
        .with_columns(
            # Each group's first row that counts in a clock (a row of
            # weight 0 counts in none).
            ts_decade=pl.col("ts")
            - pl.when(pl.Series(_first_counted(w, n))).then(DECADE).otherwise(pl.duration(days=0)),
        )
    )


def _returns(n: int) -> np.ndarray:
    """Three series of returns, `ra`, `rb` and `rc`, for `garch` and `dcc`,
    from a generator of their own (no other column's draws move): each a
    GJR-GARCH(1,1), `σ² = 0.01 + (0.10 + 0.15·1[r < 0]) r² + 0.82 σ²` on
    the previous row, whose innovations are a DCC(1,1) of `a = 0.12`, `b =
    0.85` about a correlation of 0.3. On the linear `y` and `z` neither
    model ever starts: each waits until a likelihood-ratio test tells its
    process from a constant, so every output was null (PLAN 578j, CI-2).
    The seed is one whose `garch` starts both targets before `RESUME_AT`
    and whose `dcc` reports a correlation in the second half."""
    rng = np.random.default_rng([9, 578])
    qbar = np.full((3, 3), 0.3) + 0.7 * np.eye(3)
    q, e = qbar.copy(), np.zeros(3)
    var, prev = np.full(3, 0.01 / (1 - 0.10 - 0.075 - 0.82)), np.zeros(3)
    out = np.empty((n, 3))
    for i in range(n):
        q = 0.03 * qbar + 0.12 * np.outer(e, e) + 0.85 * q
        d = np.sqrt(np.diag(q))
        e = np.linalg.cholesky(q / np.outer(d, d)) @ rng.standard_normal(3)
        var = 0.01 + (0.10 + 0.15 * (prev < 0)) * prev**2 + 0.82 * var
        prev = np.sqrt(var) * e
        out[i] = prev
    return out


#: The wide block's columns: `lasso`'s sweep in parts from 40 features, and
#: `hmm`'s factor form from 13.
WIDE = [f"v{j}" for j in range(40)]


def _wide(n: int) -> np.ndarray:
    """Forty correlated features, from a generator of their own."""
    rng = np.random.default_rng([40, 578])
    common = rng.standard_normal((n, 1))
    return 0.5 * common + rng.standard_normal((n, len(WIDE)))


def _first_counted(w: np.ndarray, n: int) -> np.ndarray:
    """True on each of the three groups' first row of weight other than 0."""
    first = np.zeros(n, dtype=bool)
    for g in range(3):
        first[np.flatnonzero((np.arange(n) % 3 == g) & (w != 0.0))[0]] = True
    return first


#: The row from which `x1b` is `x1` stepped up by 4.
BREAK_AT = 220
#: Ten years and the two leap days between 2014 and 2024, as `ts_decade`
#: moves each group's first row back.
DECADE = pl.duration(days=3653)
#: A decade in nanoseconds, 365.25 days a year: the clock age
#: `activations` holds `ts_decade`'s groups to.
DECADE_NS = int(3652.5 * 86_400 * 10**9)


#: Where `states` stops: the first half of `stream()`.
HALF = 200

CLOCK = dict(clock="t", gap_cap=6.0, weight="w")
FIT = dict(targets=["y"], features=["x0", "x1", "x2"], group="g", half_life=25.0, **CLOCK)


class _Formula:
    """A window expression as a target, built from ``po.<op>`` only when its
    spec is: the probe imports under releases that have no such operator, and
    one of those refuses the spec that names it, not the run."""

    def __init__(self, op: str, column: str, *, alias: str, **kw) -> None:
        self.op, self.column, self.alias, self.kw = op, column, alias, kw

    def build(self) -> pl.Expr:
        kw = dict(self.kw)
        if self.op.startswith("ewm"):
            kw.setdefault("half_life", INF)
        # The operators' home is `ops` (a root name of 2.0.0 warns there);
        # a release before `ops` had them at the root alone.
        home = getattr(po, "ops", po)
        fn = getattr(home, self.op, None) or getattr(po, self.op)
        return fn(self.column, reverse=True, **kw).alias(self.alias)


def _fullest() -> dict[str, dict]:
    """Each regression with every output it can emit, as
    ``tests/test_budgets_memory.py``'s ``fullest`` switches them on."""
    every = dict(
        coef_every=0,
        emit_clocks=True,
        emit_sigma=True,
        emit_zscore=True,
        emit_metrics=True,
        conformal=0.9,
        resid_quantiles=[0.1, 0.9],
        emit_autocorr=True,
        emit_drift=True,
        drift_threshold=20.0,
        emit_calibration=True,
        emit_breaks=True,
        emit_specification=True,
        emit_tails=True,
        emit_feature_health=True,
        emit_averaged=True,
        emit_error_inflation=True,
        emit_se_coef=True,
        emit_robust_se=True,
        emit_influence=True,
        emit_selected=True,
    )
    # One slot has nothing to choose or average across: both are refused
    # there (`emit_averaged` since task 278), and only lasso's path is a grid.
    one_slot = {"emit_selected", "emit_averaged"}
    simple = one_slot | {"emit_error_inflation", "emit_se_coef", "emit_robust_se"}
    simple |= {"emit_influence"}
    lacks = {
        "ewridge": one_slot,
        "rls": one_slot,
        "lasso": {"emit_error_inflation", "emit_se_coef", "emit_robust_se", "emit_influence"},
        "kalman": one_slot | {"emit_robust_se"},
        "huber": simple,
        "quantile": simple,
        "ftrl": simple,
        "sgd": simple,
        "pa": simple,
        # A glm refuses the checks of a constant, normal residual, and an
        # expectile off the mean (0.8 here) the mean's.
        "glm": simple | {"emit_specification", "emit_tails"},
        "expectile": simple | {"emit_calibration", "emit_breaks"},
        "holt": simple | {"emit_feature_health"},
    }
    fit = dict(FIT, targets=["y", "z"])
    base = {
        "ewridge": fit,
        "rls": fit,
        "lasso": dict(fit, penalties=[0.2, 0.05, 0.0]),
        "kalman": dict(fit, coef_half_life=50.0),
        "huber": fit,
        "quantile": dict(fit, quantile=0.8),
        "ftrl": dict(fit, targets=["hit"]),
        "sgd": dict(fit, learning_rate=0.01),
        "pa": fit,
        "glm": dict(fit, targets=["hit"]),
        "expectile": dict(fit, expectile=0.8),
        "holt": dict(targets=["y"], group="g", half_life=25.0, **CLOCK),
    }
    return {
        kind: {**kw, **{k: v for k, v in every.items() if k not in lacks[kind]}}
        for kind, kw in base.items()
    }


def _ew_class_window() -> dict:
    return dict(
        features=["x1", "x2"],
        label="lab",
        classes=["down", "up"],
        precision_prior=1.0,
        half_life=25.0,
        window_size=60.0,
        coef_every=0,
        **CLOCK,
    )


#: (name, builder, keyword arguments, the specs it reads, if any).
WORKLOAD: list[tuple[str, str, dict, list[str]]] = [
    ("ridge", "ewridge", dict(FIT, min_weight=4.0), []),
    ("ridge_grid", "ewridge", dict(FIT, ridge=[1e-6, 0.5], half_life=[10.0, 60.0]), []),
    (
        "ridge_diag",
        "ewridge",
        dict(
            FIT,
            targets=["y", "z"],
            emit_sigma=True,
            emit_zscore=True,
            emit_drift=True,
            # The default every release before task 168 gave a numeric
            # clock, which this build requires be said with a clock.
            drift_threshold=20.0,
            emit_metrics=True,
            emit_autocorr=True,
            resid_quantiles=[0.1, 0.9],
            conformal=0.9,
        ),
        [],
    ),
    ("ridge_window", "ewridge", dict(FIT, window_size=40.0), []),
    ("ridge_origin", "ewridge", dict(FIT, fit_intercept=False, standardize=False), []),
    (
        "ridge_session",
        "ewridge",
        dict(FIT, session="s", session_gap=10.0, session_shrink=0.5, long_half_life=200.0),
        [],
    ),
    ("ridge_delay", "ewridge", dict(FIT, embargo=3.0), []),
    ("rls", "rls", dict(FIT), []),
    ("lasso", "lasso", dict(FIT, penalties=[0.2, 0.05, 0.0]), []),
    ("enet", "lasso", dict(FIT, penalties=[0.1, 0.0], l1_ratio=0.5), []),
    ("kalman", "kalman", dict(FIT, coef_half_life=50.0), []),
    ("huber", "huber", dict(FIT), []),
    # Task 271's scales: the MAD, the default since, with a memory of its
    # own, and the RMS, every release's before it; and sgd's Huber loss.
    # A release without `huber_scale` refuses them: not comparable.
    ("huber_mad", "huber", dict(FIT, huber_scale="mad", huber_scale_half_life=10.0), []),
    ("huber_rms", "huber", dict(FIT, huber_scale="rms"), []),
    (
        "sgd_huber_mad",
        "sgd",
        dict(FIT, learning_rate=0.01, loss="huber", huber_scale="mad"),
        [],
    ),
    # Task 277's refit of the first rows, on by default at 500 rows, which
    # no group of this stream reaches: at 90 each group's buffer is part
    # full at the half and its refit runs in the second. A release without
    # `huber_refit_rows` refuses them: not comparable.
    ("huber_refit", "huber", dict(FIT, huber_refit_rows=90), []),
    (
        "sgd_huber_refit",
        "sgd",
        dict(FIT, learning_rate=0.01, loss="huber", huber_refit_rows=90),
        [],
    ),
    ("quantile", "quantile", dict(FIT, quantile=0.8), []),
    ("ftrl", "ftrl", dict(FIT, targets=["hit"]), []),
    ("sgd", "sgd", dict(FIT, learning_rate=0.01), []),
    ("pa", "pa", dict(FIT), []),
    # A count, with the refit filled on its 40th learned row of a group.
    ("glm", "glm", dict(FIT, targets=["hit"], refit_rows=40), []),
    # A Tweedie amount with no refit, every row held to the step cap.
    ("glm_tweedie_cap", "glm", dict(FIT, targets=["hit"], family="tweedie", refit_rows=0), []),
    # An upper expectile, its refit filled on its 40th learned row of a group.
    ("expectile", "expectile", dict(FIT, expectile=0.8, refit_rows=40), []),
    ("holt", "holt", dict(targets=["y"], group="g", half_life=25.0, **CLOCK), []),
    (
        "ew_cov",
        "ew_cov",
        dict(
            features=["x0", "x1", "x2"],
            stats=["mean", "var", "std", "cov", "corr"],
            group="g",
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    (
        "ew_cov_scores",
        "ew_cov",
        dict(
            features=["x1", "x2", "y"],
            stats=["partial_corr", "mahal"],
            precision_prior=1e-6,
            mahal_quantiles=[0.5, 0.99],
            pca=2,
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    (
        "ew_cov_lags",
        "ew_cov",
        dict(
            features=["x1", "x2"], stats=["corr", "lag_corr"], lags=[1, 3], half_life=25.0, **CLOCK
        ),
        [],
    ),
    ("kmeans", "kmeans", dict(features=["x1", "x2"], n_clusters=2, half_life=25.0, **CLOCK), []),
    ("micro", "micro", dict(features=["x1", "x2"], eps=0.5, half_life=25.0, **CLOCK), []),
    (
        "ew_class",
        "ew_class",
        dict(
            features=["x1", "x2"],
            label="lab",
            classes=["down", "up"],
            precision_prior=1.0,
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    ("seqtest", "seqtest", dict(targets=["y"], a="ridge", b="kalman"), ["ridge", "kalman"]),
    (
        "marginal",
        "marginal",
        dict(targets=["y"], features=["x1", "x2"], lags=[1], bins=4, half_life=25.0, **CLOCK),
        [],
    ),
    ("deco", "deco", dict(features=["x0", "x1", "x2"], half_life=25.0, **CLOCK), []),
    ("corrchange", "corrchange", dict(features=["x1", "x2"], window_rows=40, **CLOCK), []),
    ("bocpd", "bocpd", dict(features=["x1", "x2"], covariance_prior=[1.0], **CLOCK), []),
    (
        "hmm",
        "hmm",
        # `warm_rows` given: the default (1,000 rows since task 278) would
        # seed nothing in the probe's 400.
        dict(
            features=["x1", "x2"],
            n_components=2,
            precision_prior=0.1,
            half_life=25.0,
            warm_rows=50,
            **CLOCK,
        ),
        [],
    ),
    (
        "rcov",
        "rcov",
        dict(features=["x1", "x2"], group="block", group_close="monotone", block_rows=30),
        [],
    ),
    # No `weight`: an audit counts every row whatever it weighs.
    (
        "audit",
        "audit",
        dict(columns=["x0", "x1", "y"], group="g", pairs=True, clock="t", gap_cap=6.0),
        [],
    ),
    # Each target a series of returns, after a start of 50 rows: one group,
    # so each start has the rows its test needs.
    (
        "garch",
        "garch",
        dict(targets=["ra", "rb"], start_rows=50, forecast_rows=[3, 1], group="g", **CLOCK),
        [],
    ),
    # Three series' correlations, through GARCH margins, after both starts
    # of 50 rows: the asymmetric model, estimated.
    (
        "dcc",
        "dcc",
        dict(
            targets=["ra", "rb", "rc"],
            asymmetric=True,
            margin_start_rows=50,
            start_rows=50,
            half_life=50.0,
            **CLOCK,
        ),
        [],
    ),
    # A digest a target: the tails withheld until the rows identify them.
    (
        "tdigest",
        "tdigest",
        dict(targets=["y", "z"], quantiles=[0.05, 0.25, 0.95], group="g", **CLOCK),
        [],
    ),
    # A sketch a target over a row window, its tails withheld likewise.
    (
        "ddsketch",
        "ddsketch",
        dict(targets=["y", "z"], quantiles=[0.05, 0.25, 0.95], window_rows=200, group="g", **CLOCK),
        [],
    ),
    # Each target's power at two periods.
    (
        "spectral",
        "spectral",
        dict(targets=["y", "z"], periods=[8.0, 3.0], group="g", half_life=25.0, **CLOCK),
        [],
    ),
    # Each target on its own lags, the features beside them.
    (
        "ar",
        "ar",
        dict(
            targets=["y", "z"],
            features=["x0", "x1"],
            lags=[1, 2],
            group="g",
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    # Each target a monotone function of one feature, decayed, read at the
    # nearer end beyond its rows.
    (
        "isotonic",
        "isotonic",
        dict(
            targets=["y", "z"],
            features=["x0"],
            out_of_bounds="clip",
            group="g",
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    # Every target on every target's lags, a Granger test read per row.
    (
        "vector_ar",
        "vector_ar",
        dict(
            targets=["y", "z"],
            features=["x0", "x1"],
            lags=2,
            select="aic",
            causality=[{"caused": "y", "causing": "z"}],
            group="g",
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    (
        "hoeffding_tree_classifier",
        "hoeffding_tree_classifier",
        dict(
            features=["x0", "x1"],
            label="lab",
            classes=["down", "up"],
            grace_period=20.0,
            delta=1e-2,
            group="g",
            **CLOCK,
        ),
        [],
    ),
    (
        "hoeffding_tree_regressor",
        "hoeffding_tree_regressor",
        dict(
            targets=["y", "z"],
            features=["x0", "x1"],
            grace_period=20.0,
            delta=1e-2,
            group="g",
            **CLOCK,
        ),
        [],
    ),
    # Each row where `y` or `z` is above 0 an event of that node; the zero weights
    # learn none of theirs.
    (
        "hawkes",
        "hawkes",
        dict(targets=["y", "z"], kernel_decays=[0.5, 2.0], group="g", half_life=25.0, **CLOCK),
        [],
    ),
    # Each target a series of returns, decayed, with Lee and Mykland's own
    # window.
    (
        "realized",
        "realized",
        dict(targets=["y", "z"], lm_window=30, group="g", half_life=25.0, **CLOCK),
        [],
    ),
    # Two components of three columns by the incremental SVD.
    (
        "pca",
        "pca",
        dict(features=["x0", "x1", "x2"], n_components=2, group="g", half_life=25.0, **CLOCK),
        [],
    ),
    # Each target a series of returns, decayed, a term of each kind.
    (
        "power_variation",
        "power_variation",
        dict(
            targets=["y", "z"],
            terms={
                "rv": {"powers": [2]},
                "bv": {"powers": [1, 1], "normalize": True},
                "lrv": {"kernel": "parzen", "bandwidth": 5, "center": True},
                "sub": {"powers": [2], "scale": 4},
            },
            group="g",
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    # A skew, a cross-moment and an autocorrelation one clock unit back, each
    # with its standard error; the zero weights learn nothing.
    (
        "ewm_moments",
        "ewm_moments",
        dict(
            terms={
                "skew": {"x1": 3},
                "co": {"x1": 1, "x2": 2},
                "ac1": {"x1": 1, "x1@1": 1},
                "m3": {"x2": 3},
                "kurt": {"x2": 4},
            },
            kind={"skew": "standardized", "ac1": "standardized", "kurt": "standardized"},
            se=True,
            group="g",
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
]

#: The switches and paths the workload above never reached (review round 9,
#: J-2), written against the 1.0 API: every output each regression can emit,
#: a temporal clock, the blocked Gram, feature sets and a prior, the other
#: losses and modes, the windows of the other kinds, a formula target, a
#: drift reset, a solve and coefficient cadence, a restart on a step back.
#: A release before 1.0 refuses some of them by a parameter it lacks.
#: `tests/test_release_compare.py` holds the workload to every field name
#: the fullest workload emits and to the paths it names.
REACH: list[tuple[str, str, dict, list[str]]] = [
    # Every output each regression can emit, as the memory budgets'
    # workload switches them on (tests/test_budgets_memory.py).
    *[(f"full_{kind}", kind, kw, []) for kind, kw in _fullest().items()],
    (
        "ew_cov_full",
        "ew_cov",
        dict(
            features=["x0", "x1", "x2"],
            stats=["mean", "var", "std", "cov", "corr", "partial_corr", "mahal"],
            precision_prior=1e-6,
            mahal_quantiles=[0.5, 0.99],
            pca=2,
            group="g",
            half_life=25.0,
            emit_clocks=True,
            **CLOCK,
        ),
        [],
    ),
    # A temporal clock, every clock parameter a duration.
    (
        "ridge_datetime",
        "ewridge",
        dict(
            FIT,
            clock="ts",
            half_life="25s",
            gap_cap="6s",
            coef_every="20s",
            solve_every="2s",
            embargo="3s",
            session="s",
            session_gap="10s",
            emit_clocks=True,
        ),
        [],
    ),
    # The blocked Gram on two threads, the targets' rows apart.
    (
        "ridge_blocked",
        "ewridge",
        dict(
            FIT,
            targets=["y", "z"],
            gram_block_rows=16,
            gram_threads=2,
            target_gaps="pairwise",
        ),
        [],
    ),
    (
        "ridge_sets",
        "ewridge",
        dict(
            FIT,
            feature_sets={"a": ["x0", "x1"], "b": ["x1", "x2"]},
            emit_selected=True,
        ),
        [],
    ),
    ("ridge_prior", "ewridge", dict(FIT, coef_prior=[[0.4, 1.0, -0.4, 0.1]]), []),
    (
        "ridge_reset",
        "ewridge",
        dict(FIT, emit_drift=True, drift_threshold=3.0, drift_action="reset"),
        [],
    ),
    (
        "ridge_cadence",
        "ewridge",
        dict(FIT, solve_every=8.0, max_rows_between_solves=3, max_rows_between_coefs=5),
        [],
    ),
    (
        "ridge_step_back",
        "ewridge",
        dict(FIT, group=None, clock="t_back", restart_after_step_back=10.0),
        [],
    ),
    ("ridge_both", "ewridge", dict(FIT, window_size=40.0, closed="both"), []),
    ("ridge_lam", "ewridge", dict(FIT, half_life=None, lam=0.97), []),
    (
        "kalman_shared",
        "kalman",
        dict(
            FIT,
            targets=["y", "z"],
            revert_half_life=200.0,
            share_p=True,
            q=[1e-4, 1e-3, 1e-3, 1e-3],
            obs_var=0.1,
        ),
        [],
    ),
    ("kalman_standardized", "kalman", dict(FIT, coef_half_life=50.0, standardize=True), []),
    # Standardized with no `obs_var` and no intercept, each slope's prior
    # waits for its feature's scale, which the first row cannot give; a
    # `min_weight` of one row's weight lets the next row arrive before then,
    # which predicts nothing and withholds as `below_warm_rows` (1.2.0
    # predicted 0.0 there: review round 9, B-1). With an intercept the first
    # row sizes the diffuse intercept, and no row waits (task 278).
    (
        "kalman_unsized",
        "kalman",
        dict(FIT, coef_half_life=50.0, standardize=True, min_weight=1.0, fit_intercept=False),
        [],
    ),
    *[
        (f"sgd_{loss}", "sgd", dict(FIT, learning_rate=0.01, loss=loss, **extra), [])
        for loss, extra in (
            ("huber", {"huber_delta": 1.0}),
            ("quantile", {"quantile": 0.8}),
            ("epsilon_insensitive", {"eps": 0.1}),
            ("poisson", {"targets": ["hit"]}),
            ("logistic", {"targets": ["hit"]}),
        )
    ],
    ("sgd_adagrad", "sgd", dict(FIT, learning_rate=0.1, schedule="adagrad"), []),
    ("ftrl_squared", "ftrl", dict(FIT, loss="squared"), []),
    *[(f"pa_{mode}", "pa", dict(FIT, mode=mode, c=0.5), []) for mode in ("pa1", "pa2")],
    (
        "marginal_window",
        "marginal",
        dict(
            targets=["y"],
            features=["x1", "x2"],
            lags=[1, 2],
            window_lags=True,
            window_size=60.0,
            closed="both",
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    (
        "marginal_edges",
        "marginal",
        dict(
            targets=["y"],
            features=["x1", "x2"],
            bin_edges=[[-1.0, 0.0, 1.0], [-0.5, 0.5]],
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    ("lasso_window", "lasso", dict(FIT, penalties=[0.1, 0.0], window_size=40.0), []),
    ("ew_class_window", "ew_class", _ew_class_window(), []),
    # A target that is a window expression over the next 10 clock
    # units, learned once they have passed. Built when the spec is,
    # so a release without the operator refuses the spec alone.
    (
        "ridge_formula",
        "ewridge",
        dict(
            FIT,
            group=None,
            targets=[_Formula("ewm_mean", "y", window_size=10.0, alias="fwd_y")],
            embargo=10.0,
        ),
        [],
    ),
    (
        "rcov_session",
        "rcov",
        dict(
            features=["x1", "x2"],
            group="g",
            session="s",
            group_close="session",
            block_rows=30,
        ),
        [],
    ),
]

#: The paths added since 2.0.0, each a spec a release before refuses (an
#: addition) and this build runs: every drift detector, `kalman`'s square
#: root, `hmm`'s factor form and its prior's rows, `lasso`'s sweep in parts,
#: `glm`'s other families and canonical links, `expectile` and `hawkes` on
#: a window, `hawkes`' half-lives, and the cadences given as spans.
SINCE_2_0: list[tuple[str, str, dict, list[str]]] = [
    *[
        (f"ridge_drift_{kind}", "ewridge", dict(FIT, emit_drift=True, drift_kind=kind, **extra), [])
        for kind, extra in (
            ("cusum", dict(drift_threshold=20.0)),
            ("adwin", {}),
            ("ddm", dict(drift_options={"error_above": 0.5})),
            ("eddm", dict(drift_options={"error_above": 0.5})),
            ("kswin", {}),
        )
    ],
    (
        "kalman_square_root",
        "kalman",
        dict(FIT, coef_half_life=50.0, covariance_form="square_root"),
        [],
    ),
    (
        "hmm_factor",
        "hmm",
        dict(
            features=WIDE[:13],
            n_components=2,
            precision_prior=0.1,
            prior_rows=30.0,
            half_life=50.0,
            warm_rows=50,
            **CLOCK,
        ),
        [],
    ),
    ("lasso_split", "lasso", dict(FIT, features=WIDE, group=None, penalties=[0.1, 0.0]), []),
    ("glm_binomial", "glm", dict(FIT, targets=["hit"], family="binomial", refit_rows=40), []),
    (
        "glm_gamma_inverse",
        "glm",
        dict(FIT, targets=["amount"], family="gamma", link="inverse", refit_rows=40),
        [],
    ),
    (
        "glm_tweedie_power",
        "glm",
        dict(FIT, targets=["amount"], family="tweedie", link="power", refit_rows=40),
        [],
    ),
    ("glm_refit_span", "glm", dict(FIT, targets=["hit"], refit_span=60.0), []),
    (
        "expectile_window",
        "expectile",
        dict(FIT, expectile=0.8, half_life=INF, window_size=60.0),
        [],
    ),
    ("expectile_refit_span", "expectile", dict(FIT, expectile=0.8, refit_span=60.0), []),
    ("huber_refit_span", "huber", dict(FIT, huber_refit_span=90.0), []),
    (
        "hawkes_window",
        "hawkes",
        dict(
            targets=["y", "z"],
            kernel_half_lives=[1.5, 0.4],
            half_life=INF,
            window_size=80.0,
            group="g",
            **CLOCK,
        ),
        [],
    ),
    ("garch_span", "garch", dict(targets=["ra", "rb"], start_span=60.0, **CLOCK), []),
]

WORKLOAD += REACH + SINCE_2_0


def _clocked(name: str) -> tuple[str, dict]:
    """A spec of `WORKLOAD` that reads the clock `t`, as its builder and its
    keyword arguments."""
    builder, kw = next((b, kw) for n, b, kw, _ in WORKLOAD if n == name)
    return builder, dict(kw)


#: The kinds that read a clock, one spec of each from `WORKLOAD`, for the
#: paths every kind takes on its own: a reset after a step back, a session.
CLOCKED = [
    "rls", "lasso", "kalman", "huber", "quantile", "ftrl", "sgd", "pa", "glm", "expectile",
    "holt", "ew_cov",
    "kmeans", "micro", "ew_class", "marginal", "deco", "corrchange", "bocpd", "hmm",
    "garch", "hawkes",
]  # fmt: skip

#: The paths that can change numbers without a tolerance-held test seeing
#: it, beside `WORKLOAD`'s: `bocpd`'s pruning and its fold at the cap, a
#: clock at a decade's age, a reset after a step back in every kind that
#: reads a clock, and a session crossed by each kind of clock. A
#: compensated mean's low part is reached by `kmeans` and `micro` in
#: `WORKLOAD`. `activations` counts each path in the run, so the comparison
#: shows each was reached rather than assuming it; `main` also runs every
#: spec on the first rows of each group at zero weight and null, and saved,
#: loaded and continued mid-stream.
HARD_PATHS: list[tuple[str, str, dict, list[str]]] = [
    # A number `hazard` on a number clock, the clock form since PLAN 547
    # (rows before it, a declared move), and the count it read as before,
    # `hazard_rows`, which the earlier builds refuse (an addition).
    (
        "bocpd_hazard_clock",
        "bocpd",
        dict(features=["x1", "x2"], covariance_prior=[1.0], hazard=40.0, **CLOCK),
        [],
    ),
    (
        "bocpd_hazard_rows",
        "bocpd",
        dict(features=["x1", "x2"], covariance_prior=[1.0], hazard_rows=40.0, **CLOCK),
        [],
    ),
    # Pruning that drops runs from the middle of the vector, and a cap the
    # runs reach, over a break at `BREAK_AT`: the fold keeps the run
    # carrying the mass, before the break and after it.
    *[
        (
            f"bocpd_fold_{shape}",
            "bocpd",
            dict(
                features=["x1b", "x2"],
                covariance_type=shape,
                covariance_prior=[1.0],
                prune_below=1e-3,
                max_run=30,
                **CLOCK,
            ),
            [],
        )
        for shape in ("diag", "full")
    ],
    # A temporal clock whose groups began a decade before the rest of their
    # rows, every clock parameter a duration.
    (
        "decade_ridge",
        "ewridge",
        dict(
            FIT,
            clock="ts_decade",
            half_life="25s",
            gap_cap="6s",
            session="s",
            session_gap="10s",
            emit_clocks=True,
        ),
        [],
    ),
    (
        "decade_ew_cov",
        "ew_cov",
        dict(
            features=["x0", "x1", "x2"],
            stats=["mean", "var", "cov"],
            group="g",
            clock="ts_decade",
            half_life="25s",
            gap_cap="6s",
            weight="w",
            window_size="40s",
        ),
        [],
    ),
    (
        "decade_kmeans",
        "kmeans",
        dict(
            features=["x1", "x2"],
            n_clusters=2,
            group="g",
            clock="ts_decade",
            half_life="25s",
            gap_cap="6s",
            weight="w",
        ),
        [],
    ),
    # A reset after a step back, in every kind that reads a clock (`ewridge`'s
    # is `ridge_step_back` above).
    *[
        (
            f"step_back_{name}",
            _clocked(name)[0],
            dict(_clocked(name)[1], group=None, clock="t_back", restart_after_step_back=10.0),
            [],
        )
        for name in CLOCKED
    ],
    # A session crossed: a clock step replaced by `session_gap`, and a
    # session that resets the model.
    *[
        (
            f"session_{name}",
            _clocked(name)[0],
            dict(_clocked(name)[1], session="s", session_gap=10.0),
            [],
        )
        for name in ("kalman", "ew_cov", "kmeans", "micro", "bocpd", "hmm", "holt")
    ],
    ("session_reset", "ewridge", dict(FIT, session="s", session_gap="reset"), []),
]

#: Where each spec's resumed run is saved and loaded: inside the first
#: session, before `t_back` steps back at row 150 and before the gap at row
#: 250, so the file carries a running session, a clock and a restart's
#: watch to the rows after it.
RESUME_AT = 137

#: The first row of each group: rows 0, 1 and 2 of `stream()`.
FIRST_ROWS = 3


def first_rows_zero_weight(df: pl.DataFrame) -> pl.DataFrame:
    """`df` with each group's first row at weight 0: the first update of
    every model is the 0/0 a zero-weight row makes of an empty state."""
    first = pl.int_range(pl.len()) < FIRST_ROWS
    return df.with_columns(w=pl.when(first).then(0.0).otherwise(pl.col("w")))


#: The columns `first_rows_null` empties: every feature, target and label.
NULLED = ["x0", "x1", "x2", "x1b", "y", "z", "hit", "lab"]


def first_rows_null(df: pl.DataFrame) -> pl.DataFrame:
    """`df` with every feature, target and label null on each group's first
    row: the first row a model is fed is one it skips."""
    first = pl.int_range(pl.len()) < FIRST_ROWS
    return df.with_columns(pl.when(~first).then(pl.col(c)).alias(c) for c in NULLED)


#: The streams beside `stream()` every spec runs on, each spec's output
#: under the prefix, with which specs it applies to: the zero weights to
#: the specs that read `w`.
VARIANTS = [
    ("w0_", first_rows_zero_weight, lambda kw: kw.get("weight") == "w"),
    ("null0_", first_rows_null, lambda kw: True),
]


#: Task 144's renames, new name -> old, so the probe can run under a released
#: wheel that predates them (``scripts/compare_release.py``,
#: ``tests/test_released_state.py``): the workload is written in the current
#: names, and translated when the installed package lacks them.
_OLD_NAMES = {
    "half_life": "halflife",
    "long_half_life": "long_halflife",
    "coef_half_life": "coef_halflife",
    "revert_half_life": "revert_halflife",
    "select_half_life": "select_halflife",
    "trend_half_life": "trend_halflife",
    "embargo": "label_delay",
    "gap_cap": "max_dclock",
    "window_size": "window",
    "min_weight": "min_periods",
    "emit_zscore": "emit_resid_z",
    "fit_intercept": "add_intercept",
    "max_iter": "max_cd_iters",
    "tol": "cd_tol",
    "reset_on_flag": "reset",
    # Task 196's row counts, which say so in their names.
    "update_every_rows": "update_every",
    "split_merge_every_rows": "split_merge_every",
    "permute_every_rows": "permute_every",
}


def _current_names() -> bool:
    """Whether the installed package takes task 144's names: asked of a
    builder, since a checked builder's signature is ``(*args, **kwargs)``."""
    try:
        po.spec.ewridge("probe", targets=["y"], features=["x"], half_life=10.0)
    except TypeError:
        return False
    return True


#: The names of 2026-10-11, new name -> the 0.13 name, per builder, so the
#: probe runs under a released wheel that predates them (task 244). The
#: current package reads the old ones too, with a warning, but the workload
#: is written in the new.
_OLD_NAMES_IN = {
    "kmeans": {"n_clusters": "k", "init": "seed_rule"},
    "hmm": {"n_components": "k", "init": "seed_rule", "covariance_type": "covariance"},
    "ew_class": {"covariance_type": "covariance"},
    # Task 195's `delta`: `ridge` is still `ewridge`'s, so only `rls` moves.
    "rls": {"delta": "ridge"},
    "lasso": {"penalties": "lasso_path", "snapshot_every": "window_every"},
    "corrchange": {"window_rows": "span_rows"},
    "bocpd": {
        "mean_prior": "prior_mean",
        "mean_precision_prior": "prior_kappa",
        "degrees_of_freedom_prior": "prior_nu",
        "covariance_prior": "prior_scale",
    },
}


#: The words renamed in one builder's parameter, under its 1.0 name, new
#: word -> the 0.13 word (task 244). `bocpd`'s `covariance_type` was its
#: `emission` word, which `_speak` translates on its own.
_OLD_WORDS_IN = {
    "kmeans": {"init": {"k-means++": "kmeanspp"}},
    "hmm": {
        "init": {"k-means++": "kmeanspp"},
        "covariance_type": {"tied": "shared", "diag": "diagonal"},
    },
    "ew_class": {"covariance_type": {"tied": "shared", "diag": "diagonal"}},
}


def _names_of_1_0() -> bool:
    """Whether the installed package takes the names of 2026-10-11."""
    return "n_clusters" in inspect.signature(po.spec.kmeans).parameters


def _speak(builder: str, kw: dict) -> dict:
    """``kw`` in the installed package's own names."""
    if not _names_of_1_0():
        words = _OLD_WORDS_IN.get(builder, {})
        kw = {k: words[k].get(v, v) if k in words else v for k, v in kw.items()}
        if builder == "bocpd" and "covariance_type" in kw:
            shape = kw.pop("covariance_type")
            kw["emission"] = "gaussian" if shape == "full" else shape
        kw = {_OLD_NAMES_IN.get(builder, {}).get(k, k): v for k, v in kw.items()}
    if _current_names():
        return kw
    out: dict = {}
    for key, value in kw.items():
        if key == "restart_after_step_back":
            out["on_clock_reset"] = "reset_state"
            out["min_backwards_jump"] = value
        elif key == "ridge_scale":
            out["ridge_decay"] = value == "sum"
        elif key == "standardize" and builder == "sgd":
            out["scale_features"] = value
        elif key == "standardize" and builder == "pa" and value is False:
            pass  # task 195's default; before it, `pa` read its features raw
        elif key == "stats":
            # Task 196's `lag_corr`, `lagcorr` before it.
            out["stats"] = ["lagcorr" if s == "lag_corr" else s for s in value]
        else:
            out[_OLD_NAMES.get(key, key)] = value
    return out


def build(builder: str, name: str, kw: dict) -> dict:
    """The spec ``builder`` makes of ``kw`` in the installed package's names,
    a formula target built only now (`_Formula`)."""
    kw = {
        k: [t.build() if isinstance(t, _Formula) else t for t in v] if k == "targets" else v
        for k, v in kw.items()
    }
    return getattr(po.spec, builder)(name, **_speak(builder, kw))


#: The paths `activations` counts, each a number of rows, runs, values,
#: groups or specs that took it in the run: 0 is a path the workload no
#: longer reaches, which `scripts/compare_release.py` fails on.
PATHS = [
    "bocpd drops runs below prune_below",
    "bocpd folds the runs at max_run",
    "kmeans carries a compensated mean's low part",
    "micro carries a compensated mean's low part",
    "a numeric clock crosses a session",
    "a temporal clock crosses a session",
    "a temporal clock at a decade's age",
    "a reset after a step back",
    "zero-weight first rows",
    "null first rows",
    "a state saved mid-stream, loaded and continued",
    "garch predicts a variance",
    "dcc predicts a correlation",
]


def _states(bank) -> list[dict]:
    """The state of each group of the bank's last spec, as `write_json`
    gives it."""
    doc = json.loads(bank.write_json())
    return [group for _, group in doc["states"][-1]]


def _runs(group: dict) -> list[float]:
    """The lengths of the runs one group's `bocpd` holds."""
    (model,) = [m["model"]["Bocpd"] for m in group["models"]]
    return [r["len"] for r in model["runs"]]


def _low_parts(node: object) -> int:
    """The non-zero values under every key ending in `_lo`: what a
    compensated mean carries beyond its double."""
    if isinstance(node, dict):
        return sum(
            _nonzero(v) if str(k).endswith("_lo") else _low_parts(v) for k, v in node.items()
        )
    if isinstance(node, list):
        return sum(_low_parts(v) for v in node)
    return 0


def _nonzero(node: object) -> int:
    if isinstance(node, list):
        return sum(_nonzero(v) for v in node)
    return int(isinstance(node, (int, float)) and node != 0)


def _summary(bank, name: str) -> pl.DataFrame:
    return bank.summary().filter(pl.col("spec") == name)


def _after_first_rows(df: pl.DataFrame, kw: dict) -> list[int]:
    """The row of each group, as `kw` groups `df`, that follows the group's
    rows among the first `FIRST_ROWS`: rows 3, 4 and 5 grouped by `g`, row 3
    for one group."""
    group = kw.get("group")
    keys = df[group].to_list() if group else [None] * df.height
    first, seen, rows = set(keys[:FIRST_ROWS]), set(), []
    for i in range(FIRST_ROWS, df.height):
        if keys[i] in first and keys[i] not in seen:
            seen.add(keys[i])
            rows.append(i)
    return rows


def _first_rows_learned_nothing(df: pl.DataFrame, kw: dict, out: pl.DataFrame, name: str) -> bool:
    """Whether `name` learned nothing from the first rows, its `weight_sum`
    exactly 0 on each group's next row, and ends with a `weight_sum` finite
    and positive: no 0/0 of the first rows reached the state, where it
    would never wash out."""
    if "weight_sum" not in out[name].struct.fields:
        return False
    ws = out[name].struct.field("weight_sum")
    last = ws.drop_nulls()
    ends = last.len() > 0 and 0.0 < last[-1] < INF
    return ends and all(ws[i] == 0.0 for i in _after_first_rows(df, kw))


def activations(
    counts: dict[str, int],
    prefix: str,
    name: str,
    builder: str,
    kw: dict,
    bank,
    out: pl.DataFrame,
    df: pl.DataFrame,
) -> None:
    """Add what one run of `name` on `df`, the stream `prefix` names,
    reached to `counts`: the proof each path of `PATHS` ran, counted in the
    run's outputs, its summary and its state."""
    if prefix in ("w0_", "null0_"):
        # The first rows weighed nothing, or were skipped, every feature
        # null, so each group's next row is scored on a weight of exactly
        # 0. A spec with no feature (`holt`, `audit`, `seqtest`) counts a
        # row of null targets as one to predict, and is not counted.
        path = "zero-weight first rows" if prefix == "w0_" else "null first rows"
        counts[path] += _first_rows_learned_nothing(df, kw, out, name)
        return
    if prefix:
        return
    if builder == "bocpd":
        for group in _states(bank):
            lengths = _runs(group)
            # Without a prune the lengths are 0 to the longest, every one.
            counts["bocpd drops runs below prune_below"] += int(max(lengths)) + 1 - len(lengths)
        if "max_run" in kw:
            # `run_mode` reads the cap only where the capped run, which the
            # fold made, carries the most mass.
            mode = out[name].struct.field("run_mode")
            counts["bocpd folds the runs at max_run"] += int((mode == kw["max_run"] - 1).sum())
    if builder in ("garch", "dcc"):
        # A started fit: the rows with a variance (or a correlation) to read.
        stem = "variance_" if builder == "garch" else "corr_"
        path = "garch predicts a variance" if builder == "garch" else "dcc predicts a correlation"
        fields = [f for f in out[name].struct.fields if f.startswith(stem)]
        counts[path] += sum(int(out[name].struct.field(f).is_not_null().sum()) for f in fields)
    if builder in ("kmeans", "micro"):
        path = f"{builder} carries a compensated mean's low part"
        counts[path] += sum(_low_parts(group["models"]) for group in _states(bank))
    if "session" in kw and "clock" in kw:  # `rcov` closes its groups by session
        temporal = kw["clock"] in ("ts", "ts_decade")
        path = f"a {'temporal' if temporal else 'numeric'} clock crosses a session"
        counts[path] += int(_summary(bank, name)["session_changes"].sum())
    if kw.get("clock") == "ts_decade":
        counts["a temporal clock at a decade's age"] += sum(
            group["persisted"]["clock"]["exact"]["elapsed_ns"] >= DECADE_NS
            for group in _states(bank)
        )
    if "restart_after_step_back" in kw:
        counts["a reset after a step back"] += int(_summary(bank, name)["resets"].sum())


def _fields(
    columns: dict[str, pl.Series],
    label: str,
    name: str,
    out: pl.DataFrame,
    closed: pl.DataFrame,
    height: int,
) -> None:
    for field in out[name].struct.fields:
        columns[f"{label}.{field}"] = out[name].struct.field(field)
    for col in closed.columns if closed.height else []:
        columns[f"{label}.closed.{col}"] = closed[col].extend_constant(None, height - closed.height)


def _resume(specs: list[dict], name: str, df: pl.DataFrame, folder: str):
    """`specs` fitted on `df`'s rows before `RESUME_AT`, saved, loaded, and
    fitted on the rest: the outputs and closed groups of both halves, and
    the rows the loaded bank says it was fed."""
    bank = po.ModelBank(specs)
    first = bank.fit_predict(df.head(RESUME_AT)).select(name)
    closed = [bank.closed_groups()]
    path = os.path.join(folder, f"{name}.state")
    bank.save(path)
    loaded = po.ModelBank.load(path, specs=specs)
    fed = loaded.rows_fed
    second = loaded.fit_predict(df.slice(RESUME_AT)).select(name)
    closed.append(loaded.closed_groups())
    closed = [c for c in closed if c.height]
    both = pl.concat(closed, how="vertical_relaxed") if closed else pl.DataFrame()
    return pl.concat([first, second]), both, fed


def main(out_path: str, manifest_path: str) -> None:
    """Every spec of `WORKLOAD` and `HARD_PATHS` on `stream()`, on each of
    `VARIANTS`, and resumed from a state saved at `RESUME_AT`: every output
    field to `out_path`, what ran, what each spec's kind is and what each
    path of `PATHS` counted to the manifest."""
    warnings.simplefilter("ignore")
    df = stream()
    columns: dict[str, pl.Series] = {}
    ran: list[str] = []
    refused: dict[str, str] = {}
    kinds: dict[str, str] = {}
    built: dict[str, dict] = {}
    counts = dict.fromkeys(PATHS, 0)
    unmeasured: dict[str, str] = {}
    specs = WORKLOAD + HARD_PATHS

    def run(label: str, name: str, builder: str, kw: dict, frame: pl.DataFrame, go) -> None:
        kinds[label] = builder
        try:
            bank, out, closed = go()
        except Exception as e:  # a spec, or a stream, this build does not take
            refused[label] = f"{type(e).__name__}: {e}"[:300]
            return
        ran.append(label)
        _fields(columns, label, name, out, closed, df.height)
        try:
            activations(counts, label[: -len(name)], name, builder, kw, bank, out, frame)
        except Exception as e:  # an API or a state layout this build lacks
            unmeasured[label] = f"{type(e).__name__}: {e}"[:300]

    def fitted(stream_: pl.DataFrame, name: str, builder: str, kw: dict, reads: list[str]):
        def go():
            built[name] = build(builder, name, kw)
            bank = po.ModelBank([built[r] for r in reads] + [built[name]])
            out = bank.fit_predict(stream_)
            return bank, out, bank.closed_groups()

        return go

    for name, builder, kw, reads in specs:
        run(name, name, builder, kw, df, fitted(df, name, builder, kw, reads))
    for prefix, variant, applies in VARIANTS:
        frame = variant(df)
        for name, builder, kw, reads in specs:
            if applies(kw):
                go = fitted(frame, name, builder, kw, reads)
                run(prefix + name, name, builder, kw, frame, go)
    with tempfile.TemporaryDirectory() as folder:
        for name, builder, kw, reads in specs:

            def resumed(name=name, builder=builder, kw=kw, reads=reads):
                bank_specs = [built[r] for r in reads] + [build(builder, name, kw)]
                out, closed, fed = _resume(bank_specs, name, df, folder)
                counts["a state saved mid-stream, loaded and continued"] += fed == RESUME_AT
                return None, out, closed

            run(f"resume_{name}", name, builder, kw, df, resumed)
    pl.DataFrame(columns).write_parquet(out_path)
    manifest = {
        "version": po.__version__,
        "file": po.__file__,
        "ran": ran,
        "refused": refused,
        "kinds": kinds,
        "activations": counts,
        "unmeasured": unmeasured,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)


def states(out_dir: str, manifest_path: str) -> None:
    """Each spec fitted on the first half of `stream()` and saved to
    ``<out_dir>/<name>.state``, with the specs it reads in the same bank. The
    manifest names what was saved and the schema this build writes."""
    warnings.simplefilter("ignore")
    first = stream().head(HALF)
    saved: list[str] = []
    refused: dict[str, str] = {}
    built: dict[str, dict] = {}
    for name, builder, kw, reads in WORKLOAD:
        try:
            built[name] = build(builder, name, kw)
            bank = po.ModelBank([built[r] for r in reads] + [built[name]])
            bank.fit_predict(first)
            bank.save(f"{out_dir}/{name}.state")
        except Exception as e:  # a spec this build does not have
            refused[name] = f"{type(e).__name__}: {e}"[:300]
            continue
        saved.append(name)
    manifest = {
        "version": po.__version__,
        "schema": po.schema_version(),
        "saved": saved,
        "refused": refused,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)


def tables(bank: po.ModelBank, name: str) -> dict[str, pl.DataFrame]:
    """What a model keeps and a row does not show: the rows a closed group
    emits, the coefficients where the kind has them, and ``marginal``'s pairs
    table where it has them."""
    out = {"closed": bank.closed_groups()}
    try:
        out["coef"] = bank.coef(name)
    except ValueError as e:  # a kind without coefficients refuses by name
        if "coef" not in str(e):
            raise
    if name == "marginal":
        out["marginal"] = bank.marginal(name)
    return out


def resume(states_dir: str, out_dir: str, names: list[str]) -> None:
    """Each named file ``states`` wrote, loaded with its spec as this build
    makes it, and fitted on the second half of `stream()`: its rows to
    ``<out_dir>/<name>.rows.parquet`` and each of ``tables`` beside them."""
    warnings.simplefilter("ignore")
    second = stream().slice(HALF)
    work = {name: (builder, kw, reads) for name, builder, kw, reads in WORKLOAD}
    for name in names:
        reads = work[name][2]
        specs = [build(work[r][0], r, work[r][1]) for r in reads]
        specs.append(build(work[name][0], name, work[name][1]))
        bank = po.ModelBank.load(f"{states_dir}/{name}.state", specs=specs)
        bank.fit_predict(second).select(name).write_parquet(f"{out_dir}/{name}.rows.parquet")
        for table, frame in tables(bank, name).items():
            frame.write_parquet(f"{out_dir}/{name}.{table}.parquet")


if __name__ == "__main__":
    if sys.argv[1] == "--states":
        states(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "--resume":
        resume(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        main(sys.argv[1], sys.argv[2])
