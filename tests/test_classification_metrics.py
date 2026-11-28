"""Tests for streaming binary classification metrics (``rp.eval.classification_metrics``
and ``rp.eval.reliability_bins``), including bit-for-bit chunk invariance,
Rule 9 zero-weight handling, validation refusals, and 50-seed oracle ledger
grids against ``scikit-learn``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import ledger
import rheon
from rheon import polars as rp

TIER = "essential"


def _exact_reliability(
    p: np.ndarray, y: np.ndarray, w: np.ndarray | None, n_bins: int
) -> tuple[float, float, list[dict[str, Any]]]:
    """Exact reference uniform-[0, 1] reliability bins, ECE, and MCE."""
    if w is None:
        w = np.ones_like(p, dtype=np.float64)
    keep = np.isfinite(p) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    p = p[keep]
    y = y[keep]
    w = w[keep]
    pc = np.clip(p, 0.0, 1.0)
    bins = np.minimum(np.floor(pc * n_bins).astype(np.int64), n_bins - 1)
    total_w = float(np.sum(w))
    ece = 0.0
    mce = 0.0
    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        mask = bins == b
        nb = int(np.sum(mask))
        wb = float(np.sum(w[mask])) if nb else 0.0
        if wb > 0.0:
            mp = float(np.sum(w[mask] * pc[mask]) / wb)
            ma = float(np.sum(w[mask] * y[mask]) / wb)
            gap = abs(ma - mp)
            ece += (wb / total_w) * gap
            mce = max(mce, gap)
        else:
            mp = None
            ma = None
            gap = None
        rows.append(
            {
                "bin": b,
                "bin_lo": b / n_bins,
                "bin_hi": (b + 1) / n_bins,
                "n": nb,
                "weight_sum": wb,
                "mean_pred": mp,
                "mean_actual": ma,
                "abs_gap": gap,
            }
        )
    return ece, mce, rows


def _make_discrete_frame(seed: int, n: int = 400) -> pl.DataFrame:
    """Discrete probability grid with ties, varying weights, and occasional zeros."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.05, 0.95, 19)
    p = rng.choice(grid, size=n)
    y = (rng.random(n) < p).astype(np.float64)
    # Ensure both classes appear even on extreme seeds.
    if y.sum() == 0:
        y[0] = 1.0
    if y.sum() == n:
        y[0] = 0.0
    w = rng.uniform(0.25, 2.5, size=n)
    w[rng.random(n) < 0.06] = 0.0
    return pl.DataFrame({"p": p, "y": y, "w": w})


def _make_continuous_frame(seed: int, n: int = 10_000, *, imbalanced: bool = False) -> pl.DataFrame:
    """Continuous 10k-row binary stream triggering full TDigest merges."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    shift = -2.4 if imbalanced else 0.0
    logits = shift + 1.4 * z
    p_true = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p_true).astype(np.float64)
    # Slightly miscalibrated scores so ECE/MCE are non-trivial.
    p = 1.0 / (1.0 + np.exp(-(0.85 * logits + 0.15)))
    w = rng.uniform(0.5, 1.5, size=n)
    return pl.DataFrame({"p": p, "y": y, "w": w})


# ---------------------------------------------------------------------------
# Functional, spec-mode, plain-mode, Rule 9, and edge-case tests
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_plain_column_exact_parity_with_sklearn_and_bins(self) -> None:
        df = _make_discrete_frame(7, n=500)
        res = rp.eval.classification_metrics(
            df, pred="p", target="y", weight="w", n_bins=10, min_samples=1
        )
        assert res.columns == [
            "n",
            "n_pos",
            "n_neg",
            "weight_sum",
            "pos_rate",
            "roc_auc",
            "pr_auc",
            "ece",
            "mce",
            "brier",
        ]
        assert res.height == 1
        row = res.row(0, named=True)

        p = df["p"].to_numpy()
        y = df["y"].to_numpy()
        w = df["w"].to_numpy()
        pos = w > 0.0
        assert row["n"] == int(np.sum(pos))
        assert row["n_pos"] == int(np.sum(pos & (y == 1.0)))
        assert row["n_neg"] == int(np.sum(pos & (y == 0.0)))
        np.testing.assert_allclose(row["weight_sum"], float(np.sum(w[pos])), rtol=1e-13)
        np.testing.assert_allclose(
            row["pos_rate"], float(np.sum(w[pos] * y[pos]) / np.sum(w[pos])), rtol=1e-13
        )
        np.testing.assert_allclose(
            row["roc_auc"], roc_auc_score(y[pos], p[pos], sample_weight=w[pos]), atol=1e-13
        )
        np.testing.assert_allclose(
            row["pr_auc"],
            average_precision_score(y[pos], p[pos], sample_weight=w[pos]),
            atol=1e-13,
        )
        np.testing.assert_allclose(
            row["brier"], brier_score_loss(y[pos], p[pos], sample_weight=w[pos]), atol=1e-13
        )
        want_ece, want_mce, want_bins = _exact_reliability(p, y, w, 10)
        np.testing.assert_allclose(row["ece"], want_ece, atol=1e-14)
        np.testing.assert_allclose(row["mce"], want_mce, atol=1e-14)

        bins_df = rp.eval.reliability_bins(df, pred="p", target="y", weight="w", n_bins=10)
        assert bins_df.height == 10
        assert bins_df.columns == [
            "bin",
            "bin_lo",
            "bin_hi",
            "n",
            "weight_sum",
            "mean_pred",
            "mean_actual",
            "abs_gap",
        ]
        for got_b, want_b in zip(bins_df.to_dicts(), want_bins, strict=True):
            assert got_b["bin"] == want_b["bin"]
            assert got_b["n"] == want_b["n"]
            np.testing.assert_allclose(got_b["bin_lo"], want_b["bin_lo"], atol=1e-15)
            np.testing.assert_allclose(got_b["bin_hi"], want_b["bin_hi"], atol=1e-15)
            np.testing.assert_allclose(got_b["weight_sum"], want_b["weight_sum"], atol=1e-13)
            if want_b["mean_pred"] is None:
                assert got_b["mean_pred"] is None
                assert got_b["mean_actual"] is None
                assert got_b["abs_gap"] is None
            else:
                np.testing.assert_allclose(got_b["mean_pred"], want_b["mean_pred"], atol=1e-14)
                np.testing.assert_allclose(got_b["mean_actual"], want_b["mean_actual"], atol=1e-14)
                np.testing.assert_allclose(got_b["abs_gap"], want_b["abs_gap"], atol=1e-14)

    def test_spec_mode_ftrl_sgd_glm_and_bare_name(self) -> None:
        rng = np.random.default_rng(19)
        n = 400
        x0 = rng.normal(size=n)
        x1 = rng.normal(size=n)
        g = np.where(np.arange(n) % 2 == 0, "A", "B")
        logits = 1.1 * x0 - 0.8 * x1
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float64)
        df = pl.DataFrame({"g": g, "x0": x0, "x1": x1, "y": y})

        s_ftrl = rp.spec.ftrl(
            "ftrl_m", targets=["y"], features=["x0", "x1"], half_life=50.0, group="g"
        )
        s_sgd = rp.spec.sgd(
            "sgd_m",
            targets=["y"],
            features=["x0", "x1"],
            loss="logistic",
            half_life=50.0,
            group="g",
        )
        s_glm = rp.spec.glm(
            "glm_m",
            targets=["y"],
            features=["x0", "x1"],
            family="binomial",
            half_life=50.0,
            group="g",
        )
        out = rp.ModelBank([s_ftrl, s_sgd, s_glm]).fit_predict(df)

        for spec in (s_ftrl, s_sgd, s_glm):
            by_dict = rp.eval.classification_metrics(out, spec, group="g", min_samples=10)
            by_name = rp.eval.classification_metrics(out, spec["name"], group="g", min_samples=10)
            assert by_dict.equals(by_name, null_equal=True)
            assert by_dict.columns[:3] == ["slot", "target", "g"]
            assert by_dict.height == 2
            assert all(0.5 < v <= 1.0 for v in by_dict["roc_auc"].to_list())
            assert all(0.0 <= v <= 1.0 for v in by_dict["ece"].to_list())

            b_dict = rp.eval.reliability_bins(out, spec, group="g", n_bins=5)
            b_name = rp.eval.reliability_bins(out, spec["name"], group="g", n_bins=5)
            assert b_dict.equals(b_name, null_equal=True)
            assert b_dict.height == 10

    def test_boolean_target_and_multi_group_keys(self) -> None:
        rng = np.random.default_rng(31)
        n = 300
        p = rng.uniform(0.0, 1.0, size=n)
        yb = rng.random(n) < p
        df = pl.DataFrame(
            {
                "g1": np.where(np.arange(n) < 150, "x", "y"),
                "g2": np.arange(n) % 3,
                "p": p,
                "y": yb,
            }
        )
        got = rp.eval.classification_metrics(
            df, pred="p", target="y", group=["g1", "g2"], min_samples=1
        )
        assert got.height == 6
        assert got.columns[:2] == ["g1", "g2"]
        bins = rp.eval.reliability_bins(df, pred="p", target="y", group=["g1", "g2"], n_bins=4)
        assert bins.height == 24

    def test_rule_9_zero_weight_rows_are_completely_ignored(self) -> None:
        df = _make_discrete_frame(23, n=300)
        # Keep only positive-weight rows as clean baseline.
        clean = df.filter(pl.col("w") > 0.0).with_columns(pl.lit("A").alias("g"))
        # Inject zero-weight rows with extreme predictions, non-binary labels, and a phantom group.
        zeros = pl.DataFrame(
            {
                "p": [0.99, -5.0, float("nan"), 0.5, 0.1],
                "y": [1.0, 99.0, 0.0, 2.5, 0.0],
                "w": [0.0, 0.0, 0.0, 0.0, 0.0],
                "g": ["A", "A", "A", "PHANTOM", "PHANTOM"],
            }
        )
        mixed = pl.concat([clean.slice(0, 100), zeros, clean.slice(100)])

        cm_clean = rp.eval.classification_metrics(
            clean, pred="p", target="y", group="g", weight="w", min_samples=1
        )
        cm_mixed = rp.eval.classification_metrics(
            mixed, pred="p", target="y", group="g", weight="w", min_samples=1
        )
        assert cm_clean.equals(cm_mixed, null_equal=True)
        assert cm_mixed["g"].to_list() == ["A"]

        rb_clean = rp.eval.reliability_bins(
            clean, pred="p", target="y", group="g", weight="w", n_bins=10
        )
        rb_mixed = rp.eval.reliability_bins(
            mixed, pred="p", target="y", group="g", weight="w", n_bins=10
        )
        assert rb_clean.equals(rb_mixed, null_equal=True)

    def test_chunk_invariance_bit_for_bit_dataframe_and_lazyframe(self) -> None:
        # Use 4,000 rows so both positive and negative TDigests perform merges.
        df = _make_continuous_frame(101, n=4_000).with_columns(
            pl.Series("g", ["g0", "g1", "g2", "g3"] * 1_000)
        )
        lf = df.lazy()

        cm_1 = rp.eval.classification_metrics(
            df, pred="p", target="y", group="g", weight="w", chunk_size=4_000, min_samples=1
        )
        cm_100 = rp.eval.classification_metrics(
            df, pred="p", target="y", group="g", weight="w", chunk_size=40, min_samples=1
        )
        cm_lazy_1 = rp.eval.classification_metrics(
            lf, pred="p", target="y", group="g", weight="w", chunk_size=4_000, min_samples=1
        )
        cm_lazy_100 = rp.eval.classification_metrics(
            lf, pred="p", target="y", group="g", weight="w", chunk_size=40, min_samples=1
        )
        assert cm_1.equals(cm_100, null_equal=True)
        assert cm_1.equals(cm_lazy_1, null_equal=True)
        assert cm_1.equals(cm_lazy_100, null_equal=True)

        # Also verify exact IEEE-754 bit equality for every float column.
        for col in ("weight_sum", "pos_rate", "roc_auc", "pr_auc", "ece", "mce", "brier"):
            a = cm_1[col].to_numpy().view(np.uint64)
            b = cm_100[col].to_numpy().view(np.uint64)
            c = cm_lazy_100[col].to_numpy().view(np.uint64)
            assert np.array_equal(a, b), f"DataFrame bit mismatch on {col}"
            assert np.array_equal(a, c), f"LazyFrame bit mismatch on {col}"

        rb_1 = rp.eval.reliability_bins(
            df, pred="p", target="y", group="g", weight="w", n_bins=10, chunk_size=4_000
        )
        rb_100 = rp.eval.reliability_bins(
            df, pred="p", target="y", group="g", weight="w", n_bins=10, chunk_size=40
        )
        rb_lazy_100 = rp.eval.reliability_bins(
            lf, pred="p", target="y", group="g", weight="w", n_bins=10, chunk_size=40
        )
        assert rb_1.equals(rb_100, null_equal=True)
        assert rb_1.equals(rb_lazy_100, null_equal=True)

    def test_edge_cases_single_class_ties_min_samples_empty(self) -> None:
        # Empty frame -> empty schema-correct DataFrame.
        empty = pl.DataFrame(schema={"p": pl.Float64, "y": pl.Float64, "g": pl.String})
        cm_e = rp.eval.classification_metrics(empty, pred="p", target="y", group="g")
        assert cm_e.height == 0
        assert cm_e.schema["roc_auc"] == pl.Float64
        rb_e = rp.eval.reliability_bins(empty, pred="p", target="y", group="g", n_bins=5)
        assert rb_e.height == 0
        assert rb_e.schema["bin"] == pl.Int64

        # Single class -> roc_auc and pr_auc are null; ECE/MCE/Brier remain finite.
        only_pos = pl.DataFrame({"p": [0.7, 0.8, 0.9], "y": [1.0, 1.0, 1.0]})
        s = rp.eval.classification_metrics(only_pos, pred="p", target="y", min_samples=1).row(
            0, named=True
        )
        assert s["roc_auc"] is None
        assert s["pr_auc"] is None
        assert s["pos_rate"] == 1.0
        assert s["ece"] is not None and math.isfinite(s["ece"])

        # min_samples thinning keeps counts/weight_sum and nulls derived scores.
        thin = rp.eval.classification_metrics(only_pos, pred="p", target="y", min_samples=10).row(
            0, named=True
        )
        assert thin["n"] == 3
        assert thin["n_pos"] == 3
        assert thin["n_neg"] == 0
        assert thin["weight_sum"] == 3.0
        for col in ("pos_rate", "roc_auc", "pr_auc", "ece", "mce", "brier"):
            assert thin[col] is None

        # All-tied scores -> ROC-AUC = 0.5, PR-AUC = pos_rate.
        tied = pl.DataFrame({"p": [0.4] * 20, "y": [1.0] * 5 + [0.0] * 15})
        t_row = rp.eval.classification_metrics(tied, pred="p", target="y", min_samples=1).row(
            0, named=True
        )
        np.testing.assert_allclose(t_row["roc_auc"], 0.5, atol=1e-15)
        np.testing.assert_allclose(t_row["pr_auc"], 0.25, atol=1e-15)

        # Out-of-[0, 1] probabilities still work (clipped for bins, raw for Brier).
        oob = pl.DataFrame({"p": [-0.2, 1.3], "y": [0.0, 1.0]})
        oob_cm = rp.eval.classification_metrics(oob, pred="p", target="y", min_samples=1).row(
            0, named=True
        )
        assert oob_cm["roc_auc"] == 1.0
        assert oob_cm["pr_auc"] == 1.0
        np.testing.assert_allclose(oob_cm["brier"], 0.5 * (0.04 + 0.09), atol=1e-15)
        np.testing.assert_allclose(oob_cm["ece"], 0.0, atol=1e-15)

    def test_validation_and_refusals(self) -> None:
        df = _make_discrete_frame(5, n=50)
        # Non-binary target raises ValueError.
        bad_y = df.with_columns(pl.lit(0.5).alias("y"))
        with pytest.raises(ValueError, match="0/1"):
            rp.eval.classification_metrics(bad_y, pred="p", target="y", min_samples=1)
        with pytest.raises(ValueError, match="0/1"):
            rp.eval.reliability_bins(bad_y, pred="p", target="y")

        # Mode conflicts and missing arguments.
        with pytest.raises(ValueError, match="either spec, or both pred and target"):
            rp.eval.classification_metrics(df)
        with pytest.raises(ValueError, match="not both"):
            rp.eval.classification_metrics(df, "m", pred="p", target="y")
        with pytest.raises(rheon.ColumnNotFoundError):
            rp.eval.classification_metrics(df, pred="missing", target="y")
        with pytest.raises(rheon.WrongTypeError):
            rp.eval.classification_metrics(
                df.with_columns(pl.lit("abc").alias("s")), pred="s", target="y"
            )

        # Parameter bounds and types.
        with pytest.raises(ValueError, match="n_bins"):
            rp.eval.classification_metrics(df, pred="p", target="y", n_bins=0)
        with pytest.raises(rheon.WrongTypeError, match="n_bins"):
            rp.eval.classification_metrics(df, pred="p", target="y", n_bins=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="compression"):
            rp.eval.classification_metrics(df, pred="p", target="y", compression=5.0)
        with pytest.raises(ValueError, match="max_state_bytes"):
            rp.eval.classification_metrics(df, pred="p", target="y", max_state_bytes=1024)
        with pytest.raises(ValueError, match="chunk_size"):
            rp.eval.classification_metrics(df, pred="p", target="y", chunk_size=0)

        # Renamed keywords refused.
        with pytest.raises(TypeError, match="group"):
            rp.eval.classification_metrics(df, pred="p", target="y", by="w")  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="min_samples"):
            rp.eval.classification_metrics(df, pred="p", target="y", min_obs=10)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="spec"):
            rp.eval.classification_metrics(df, spec_name="m")  # type: ignore[call-arg]

        # LazyFrame refuses bad arguments BEFORE execution.
        with pytest.raises(ValueError, match="n_bins"):
            rp.eval.reliability_bins(df.lazy(), pred="p", target="y", n_bins=0)
        with pytest.raises(rheon.ColumnNotFoundError):
            rp.eval.classification_metrics(df.lazy(), pred="nope", target="y")


# ---------------------------------------------------------------------------
# 50-seed third-party oracle ledger grids vs scikit-learn
# ---------------------------------------------------------------------------


def _discrete_guard_outputs() -> list[dict[str, Any]]:
    out = []
    for seed in range(5):
        df = _make_discrete_frame(seed, n=400)
        res = rp.eval.classification_metrics(
            df, pred="p", target="y", weight="w", n_bins=10, min_samples=1
        )
        out.append(res.row(0, named=True))
    return out


def _continuous_guard_outputs() -> list[dict[str, Any]]:
    out = []
    for seed in range(5):
        for imb in (False, True):
            df = _make_continuous_frame(seed, n=10_000, imbalanced=imb)
            res = rp.eval.classification_metrics(
                df.lazy(),
                pred="p",
                target="y",
                weight="w",
                n_bins=10,
                compression=200.0,
                chunk_size=1_000,
                min_samples=1,
            )
            out.append(res.row(0, named=True))
    return out


ledger.guard(
    "discrete_exact",
    _discrete_guard_outputs,
    oracles=["scikit-learn"],
    bits=52,
)

ledger.guard(
    "continuous_stream",
    _continuous_guard_outputs,
    oracles=["scikit-learn"],
)


@pytest.mark.oracle_grid(reason="50-seed exact discrete classification metrics vs scikit-learn")
def test_fifty_seeds_discrete_exact_vs_sklearn() -> None:
    roc_errs: list[float] = []
    ap_errs: list[float] = []
    brier_errs: list[float] = []
    ece_errs: list[float] = []
    mce_errs: list[float] = []

    for seed in range(50):
        df = _make_discrete_frame(seed, n=400)
        got = rp.eval.classification_metrics(
            df, pred="p", target="y", weight="w", n_bins=10, min_samples=1
        ).row(0, named=True)
        p = df["p"].to_numpy()
        y = df["y"].to_numpy()
        w = df["w"].to_numpy()
        pos = w > 0.0

        want_roc = float(roc_auc_score(y[pos], p[pos], sample_weight=w[pos]))
        want_ap = float(average_precision_score(y[pos], p[pos], sample_weight=w[pos]))
        want_brier = float(brier_score_loss(y[pos], p[pos], sample_weight=w[pos]))
        want_ece, want_mce, _ = _exact_reliability(p, y, w, 10)

        roc_errs.append(abs(got["roc_auc"] - want_roc))
        ap_errs.append(abs(got["pr_auc"] - want_ap))
        brier_errs.append(abs(got["brier"] - want_brier))
        ece_errs.append(abs(got["ece"] - want_ece))
        mce_errs.append(abs(got["mce"] - want_mce))

    worst_roc, _ = ledger.record("discrete_exact", "roc_auc_err", roc_errs, floor=1e-12)
    worst_ap, _ = ledger.record("discrete_exact", "pr_auc_err", ap_errs, floor=1e-12)
    worst_brier, _ = ledger.record("discrete_exact", "brier_err", brier_errs, floor=1e-12)
    worst_ece, _ = ledger.record("discrete_exact", "ece_err", ece_errs, floor=1e-12)
    worst_mce, _ = ledger.record("discrete_exact", "mce_err", mce_errs, floor=1e-12)

    assert worst_roc < 1e-12, worst_roc
    assert worst_ap < 1e-12, worst_ap
    assert worst_brier < 1e-12, worst_brier
    assert worst_ece < 1e-12, worst_ece
    assert worst_mce < 1e-12, worst_mce


@pytest.mark.oracle_grid(reason="50-seed 10k-row streaming classification metrics vs scikit-learn")
def test_fifty_seeds_continuous_10k_stream_vs_sklearn() -> None:
    bal_roc_errs: list[float] = []
    bal_ap_errs: list[float] = []
    imb_roc_errs: list[float] = []
    imb_ap_errs: list[float] = []
    ece_errs: list[float] = []
    brier_errs: list[float] = []

    for seed in range(50):
        for imb in (False, True):
            df = _make_continuous_frame(seed, n=10_000, imbalanced=imb)
            got = rp.eval.classification_metrics(
                df.lazy(),
                pred="p",
                target="y",
                weight="w",
                n_bins=10,
                compression=200.0,
                chunk_size=1_000,
                min_samples=1,
            ).row(0, named=True)
            p = df["p"].to_numpy()
            y = df["y"].to_numpy()
            w = df["w"].to_numpy()

            want_roc = float(roc_auc_score(y, p, sample_weight=w))
            want_ap = float(average_precision_score(y, p, sample_weight=w))
            want_brier = float(brier_score_loss(y, p, sample_weight=w))
            want_ece, _, _ = _exact_reliability(p, y, w, 10)

            r_err = abs(got["roc_auc"] - want_roc)
            a_err = abs(got["pr_auc"] - want_ap)
            if imb:
                imb_roc_errs.append(r_err)
                imb_ap_errs.append(a_err)
            else:
                bal_roc_errs.append(r_err)
                bal_ap_errs.append(a_err)
            ece_errs.append(abs(got["ece"] - want_ece))
            brier_errs.append(abs(got["brier"] - want_brier))

    w_bal_roc, _ = ledger.record("continuous_stream", "balanced_roc_auc_err", bal_roc_errs)
    w_bal_ap, _ = ledger.record("continuous_stream", "balanced_pr_auc_err", bal_ap_errs)
    w_imb_roc, _ = ledger.record("continuous_stream", "imbalanced_roc_auc_err", imb_roc_errs)
    w_imb_ap, _ = ledger.record("continuous_stream", "imbalanced_pr_auc_err", imb_ap_errs)
    w_ece, _ = ledger.record("continuous_stream", "ece_err", ece_errs, floor=1e-12)
    w_brier, _ = ledger.record("continuous_stream", "brier_err", brier_errs, floor=1e-12)

    assert w_bal_roc < 2e-3, w_bal_roc
    assert w_bal_ap < 1e-2, w_bal_ap
    assert w_imb_roc < 2e-3, w_imb_roc
    assert w_imb_ap < 1e-2, w_imb_ap
    assert w_ece < 1e-12, w_ece
    assert w_brier < 1e-12, w_brier
