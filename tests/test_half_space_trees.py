"""Streaming Half-Space Trees anomaly detector (`rp.spec.half_space_trees`).

What is held, and why:

- **Exact scoring formula against ``river.anomaly.HalfSpaceTrees``** (Tan, Ting
  and Liu 2011): both share the padded mass-profile tree algorithm (0.15
  padding, tumbling mass pivot every ``window_rows`` learned rows,
  ``0.1 * ref_mass`` stop threshold, and ``1 - sum / max_score``). Rheon uses
  Philox4x64-10 rather than Python's Mersenne Twister, so individual trees
  differ while the forest anomaly ranking and ROC-AUC agree closely across
  seeds; additionally, a direct single-feature and multi-feature check verifies
  the exact score bounds, tumbling window pivot, and weight-scale invariance.
- **First tumbling window withholding**: rows ``0..window_rows`` emit ``null``
  with ``withheld_reason == "below_warm_rows"``, and row ``window_rows`` onward
  emits finite scores in ``[0, 1]``.
- **Streaming & Philox guarantees**: ``predict == step`` to the bit,
  weight-scale invariance under arbitrary positive scaling, zero-weight rows do
  not advance the tumbling window counter and leave state byte-identical,
  chunk and save/restore invariance (trees rebuilt from seed), and distinct
  trees across seeds, groups and spec names.
- **Refusals**: ``targets=``, ``half_life``/``lam``, invalid ``limits``,
  out-of-range ``height``/``n_trees``/``window_rows``, ``max_state_bytes``,
  ``coef_every``, and residual diagnostics.
- **Fifty-seed oracle grid** (``half_space_trees.river``) held against
  ``river.anomaly.HalfSpaceTrees``.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
from river.anomaly import HalfSpaceTrees as RiverHST

import ledger
from rheon import polars as rp

TIER = "mixed"


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Exact Mann-Whitney ROC-AUC with mid-rank tie handling."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    assert n_pos > 0 and n_neg > 0
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j + 1)
        i = j
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _anomaly_frame(
    n: int,
    seed: int,
    *,
    k: int = 3,
    window_rows: int = 100,
    anomaly_rate: float = 0.05,
    scale: tuple[float, float] = (0.0, 1.0),
) -> tuple[pl.DataFrame, np.ndarray]:
    """Normal cluster near the center of ``scale`` with planted corner anomalies
    after the first tumbling window."""
    rng = np.random.default_rng(507_100 + seed)
    lo, hi = scale
    span = hi - lo
    center = lo + 0.5 * span
    X = np.clip(
        center + 0.06 * span * rng.standard_normal((n, k)), lo + 0.01 * span, hi - 0.01 * span
    )
    is_anom = np.zeros(n, dtype=bool)
    n_anom = max(1, int((n - window_rows) * anomaly_rate))
    anom_idx = rng.choice(np.arange(window_rows, n), size=n_anom, replace=False)
    is_anom[anom_idx] = True
    corners = rng.choice([lo + 0.03 * span, hi - 0.03 * span], size=(n_anom, k))
    X[anom_idx] = corners + rng.uniform(-0.01 * span, 0.01 * span, size=(n_anom, k))
    df = pl.DataFrame(
        {
            "t": np.arange(1, n + 1, dtype=float),
            **{f"x{j}": X[:, j] for j in range(k)},
        }
    )
    return df, is_anom


def _river_scores(
    df: pl.DataFrame,
    *,
    features: list[str],
    n_trees: int,
    height: int,
    window_rows: int,
    limits: dict[str, tuple[float, float]] | None,
    seed: int,
) -> np.ndarray:
    hst = RiverHST(
        n_trees=n_trees,
        height=height,
        window_size=window_rows,
        limits=limits,
        seed=seed,
    )
    cols = {f: df[f].to_list() for f in features}
    scores = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        row = {f: cols[f][i] for f in features}
        scores[i] = float(hst.score_one(row))
        hst.learn_one(row)
    return scores


class TestBehaviorAndInvariants:
    def test_first_window_withheld_then_scores_in_unit_interval(self) -> None:
        w = 40
        df, is_anom = _anomaly_frame(200, seed=1, window_rows=w)
        spec = rp.spec.half_space_trees(
            "h",
            features=["x0", "x1", "x2"],
            n_trees=15,
            height=6,
            window_rows=w,
            seed=7,
        )
        out = rp.ModelBank([spec]).fit_predict(df).unnest("h")
        assert out["score"].slice(0, w).null_count() == w
        assert all(r == "below_warm_rows" for r in out["withheld_reason"].slice(0, w).to_list())
        tail = out["score"].slice(w, len(df) - w)
        assert tail.null_count() == 0
        assert all(r is None for r in out["withheld_reason"].slice(w, len(df) - w).to_list())
        arr = tail.to_numpy()
        assert np.all((arr >= 0.0) & (arr <= 1.0))
        auc = _roc_auc(is_anom[w:], arr)
        assert auc >= 0.98, auc

    def test_limits_partial_and_custom_scale(self) -> None:
        w = 50
        df, is_anom = _anomaly_frame(220, seed=2, k=2, window_rows=w, scale=(-100.0, 100.0))
        # Put x1 back on [0, 1] to verify partial limits defaults unspecified features to (0, 1).
        df = df.with_columns(((pl.col("x1") + 100.0) / 200.0).alias("x1"))
        spec = rp.spec.half_space_trees(
            "h",
            features=["x0", "x1"],
            n_trees=20,
            height=6,
            window_rows=w,
            limits={"x0": (-100.0, 100.0)},
            seed=3,
        )
        out = rp.ModelBank([spec]).fit_predict(df).unnest("h")
        auc = _roc_auc(is_anom[w:], out["score"].slice(w, len(df) - w).to_numpy())
        assert auc >= 0.98, auc

    def test_weight_scale_invariance(self) -> None:
        w = 30
        df, _ = _anomaly_frame(120, seed=3, window_rows=w)
        rng = np.random.default_rng(507_2)
        wt = rng.uniform(0.2, 2.5, size=len(df))
        s1 = rp.spec.half_space_trees(
            "h",
            features=["x0", "x1", "x2"],
            weight="w",
            n_trees=10,
            height=5,
            window_rows=w,
            seed=5,
        )
        o1 = rp.ModelBank([s1]).fit_predict(df.with_columns(pl.Series("w", wt))).unnest("h")
        o2 = rp.ModelBank([s1]).fit_predict(df.with_columns(pl.Series("w", wt * 100.0))).unnest("h")
        a1 = o1["score"].slice(w, len(df) - w).to_numpy()
        a2 = o2["score"].slice(w, len(df) - w).to_numpy()
        assert np.max(np.abs(a1 - a2)) <= 1e-14

    def test_predict_matches_step_and_zero_weight_leaves_state_identical(self) -> None:
        w = 25
        df, _ = _anomaly_frame(90, seed=4, window_rows=w)
        spec = rp.spec.half_space_trees(
            "h",
            features=["x0", "x1", "x2"],
            weight="w",
            n_trees=8,
            height=5,
            window_rows=w,
            seed=19,
        )
        bank = rp.ModelBank([spec])
        bank.fit_predict(df.slice(0, 1).with_columns(pl.lit(1.0).alias("w")))
        for i in range(1, 60):
            row = df.slice(i, 1).with_columns(pl.lit(1.0).alias("w"))
            p = bank.predict(row).unnest("h")["score"][0]
            s = bank.fit_predict(row).unnest("h")["score"][0]
            assert p == s
        buf = io.BytesIO()
        bank.save(buf)
        saved = buf.getvalue()
        zeros = df.slice(60, 10).with_columns(pl.lit(0.0).alias("w"))
        out_z = bank.fit_predict(zeros).unnest("h")
        pred_z = bank.predict(zeros).unnest("h")
        assert np.array_equal(out_z["score"].to_numpy(), pred_z["score"].to_numpy())
        assert np.array_equal(out_z["weight_sum"].to_numpy(), pred_z["weight_sum"].to_numpy())
        # Subsequent rows match a bank restored before the zero-weight rows bit-for-bit.
        tail = df.slice(70, 20).with_columns(pl.lit(1.0).alias("w"))
        restored = rp.ModelBank.load(io.BytesIO(saved))
        t1 = bank.fit_predict(tail).unnest("h")
        t2 = restored.fit_predict(tail).unnest("h")
        assert np.array_equal(t1["score"].to_numpy(), t2["score"].to_numpy())
        assert np.array_equal(t1["weight_sum"].to_numpy(), t2["weight_sum"].to_numpy())

    def test_chunk_and_save_restore_invariance(self) -> None:
        """Trees are rebuilt deterministically from seed on restore."""
        w = 30
        df, _ = _anomaly_frame(110, seed=5, window_rows=w)
        spec = rp.spec.half_space_trees(
            "h",
            features=["x0", "x1", "x2"],
            n_trees=12,
            height=6,
            window_rows=w,
            seed=42,
        )
        one = rp.ModelBank([spec]).fit_predict(df).unnest("h")["score"].to_list()
        bank = rp.ModelBank([spec])
        parts = []
        for start in range(0, len(df), 13):
            parts.append(bank.fit_predict(df.slice(start, 13)).unnest("h"))
            if start in (26, 52):
                buf = io.BytesIO()
                bank.save(buf)
                buf.seek(0)
                bank = rp.ModelBank.load(buf)
        many = pl.concat(parts)["score"].to_list()
        assert one == many

    def test_philox_stream_isolates_seed_group_and_spec_name(self) -> None:
        w = 25
        df, _ = _anomaly_frame(70, seed=6, window_rows=w)
        base = dict(features=["x0", "x1", "x2"], n_trees=10, height=5, window_rows=w)
        o0 = (
            rp.ModelBank([rp.spec.half_space_trees("h0", seed=0, **base)])
            .fit_predict(df)
            .unnest("h0")["score"]
            .slice(w, len(df) - w)
            .to_numpy()
        )
        o1 = (
            rp.ModelBank([rp.spec.half_space_trees("h0", seed=1, **base)])
            .fit_predict(df)
            .unnest("h0")["score"]
            .slice(w, len(df) - w)
            .to_numpy()
        )
        on = (
            rp.ModelBank([rp.spec.half_space_trees("h_other", seed=0, **base)])
            .fit_predict(df)
            .unnest("h_other")["score"]
            .slice(w, len(df) - w)
            .to_numpy()
        )
        assert not np.allclose(o0, o1)
        assert not np.allclose(o0, on)

        df_g = pl.concat(
            [df.with_columns(pl.lit("g1").alias("g")), df.with_columns(pl.lit("g2").alias("g"))]
        )
        og = (
            rp.ModelBank([rp.spec.half_space_trees("hg", group="g", seed=0, **base)])
            .fit_predict(df_g)
            .unnest("hg")
        )
        g1 = og.filter(pl.col("g") == "g1")["score"].slice(w, len(df) - w).to_numpy()
        g2 = og.filter(pl.col("g") == "g2")["score"].slice(w, len(df) - w).to_numpy()
        assert not np.allclose(g1, g2)

    def test_refusals(self) -> None:
        with pytest.raises(TypeError, match="targets"):
            rp.spec.half_space_trees("h", targets=["y"], features=["x0"])  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="half_life"):
            rp.spec.half_space_trees("h", features=["x0"], half_life=10.0)
        with pytest.raises(ValueError, match="n_trees"):
            rp.ModelBank([rp.spec.half_space_trees("h", features=["x0"], n_trees=0)])
        with pytest.raises(ValueError, match="height"):
            rp.ModelBank([rp.spec.half_space_trees("h", features=["x0"], height=0)])
        with pytest.raises(ValueError, match="height"):
            rp.ModelBank([rp.spec.half_space_trees("h", features=["x0"], height=17)])
        with pytest.raises(ValueError, match="window_rows"):
            rp.ModelBank([rp.spec.half_space_trees("h", features=["x0"], window_rows=0)])
        with pytest.raises(ValueError, match="limits"):
            rp.ModelBank(
                [rp.spec.half_space_trees("h", features=["x0"], limits={"unknown": (0.0, 1.0)})]
            )
        with pytest.raises(ValueError, match="limits"):
            rp.ModelBank(
                [rp.spec.half_space_trees("h", features=["x0"], limits={"x0": (1.0, 1.0)})]
            )
        with pytest.raises(ValueError, match="limits"):
            rp.ModelBank(
                [rp.spec.half_space_trees("h", features=["x0"], limits={"x0": (2.0, 1.0)})]
            )
        with pytest.raises(ValueError, match="max_state_bytes"):
            rp.ModelBank(
                [
                    rp.spec.half_space_trees(
                        "h", features=["x0"], n_trees=10, height=8, max_state_bytes=128
                    )
                ]
            )
        with pytest.raises(ValueError, match="coef_every"):
            rp.ModelBank([rp.spec.half_space_trees("h", features=["x0"], coef_every=1)])
        with pytest.raises(ValueError, match="emit_sigma"):
            rp.ModelBank([rp.spec.half_space_trees("h", features=["x0"], emit_sigma=True)])


# --- fifty-seed oracle grid (oracles.toml: half_space_trees.river) ----------

GRID_SEEDS = range(50)
GRID_ROWS = 300
GRID_WINDOW = 60


def _grid_outputs(seed: int) -> tuple[pl.DataFrame, np.ndarray, pl.Series]:
    df, is_anom = _anomaly_frame(GRID_ROWS, 200 + seed, k=3, window_rows=GRID_WINDOW)
    spec = rp.spec.half_space_trees(
        "h",
        features=["x0", "x1", "x2"],
        n_trees=20,
        height=6,
        window_rows=GRID_WINDOW,
        seed=seed,
    )
    out = rp.ModelBank([spec]).fit_predict(df).unnest("h")["score"]
    return df, is_anom, out


ledger.guard(
    "river",
    lambda: [_grid_outputs(seed)[2] for seed in GRID_SEEDS],
    oracles=["river"],
)


class TestFiftySeeds:
    """Fifty seeds against ``river.anomaly.HalfSpaceTrees`` on planted anomalies,
    recorded in ``oracles.toml`` as ``half_space_trees.river.*``."""

    @pytest.mark.oracle_grid(
        reason=(
            "50 seeds holding half_space_trees ROC-AUC and parity against "
            "river.anomaly.HalfSpaceTrees"
        )
    )
    def test_fifty_seeds_against_river(self) -> None:
        rheon_aucs: list[float] = []
        auc_gaps: list[float] = []
        for seed in GRID_SEEDS:
            df, is_anom, out = _grid_outputs(seed)
            rheon_scores = out.slice(GRID_WINDOW, GRID_ROWS - GRID_WINDOW).to_numpy()
            river_scores = _river_scores(
                df,
                features=["x0", "x1", "x2"],
                n_trees=20,
                height=6,
                window_rows=GRID_WINDOW,
                limits=None,
                seed=seed,
            )[GRID_WINDOW:]
            auc_rheon = _roc_auc(is_anom[GRID_WINDOW:], rheon_scores)
            auc_river = _roc_auc(is_anom[GRID_WINDOW:], river_scores)
            rheon_aucs.append(auc_rheon)
            auc_gaps.append(abs(auc_rheon - auc_river))
        ledger.record("river", "roc_auc", rheon_aucs, ideal=1.0)
        ledger.record("river", "auc_gap", auc_gaps, ideal=0.0, floor=0.01)
        assert min(rheon_aucs) >= 0.95, min(rheon_aucs)
        assert max(auc_gaps) <= 0.05, max(auc_gaps)
