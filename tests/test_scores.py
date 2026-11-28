"""Tests for streaming evaluation scores (`emit_metrics`, `metric_quantiles`)
and post-run evaluation helpers (`rp.eval.metrics`, `rp.eval.window_metrics`,
`rp.eval.interval_metrics`).

Verifies:
1. Exact legacy parity (`emit_metrics=True` == `["ic", "r2", "hit_rate"]` to 0 ulp).
2. Streaming `mae`, `pinball` (model quantile and explicit `metric_quantiles`),
   `log_loss` (binary probability models), and conformal `coverage`, `width`,
   `winkler` against `scikit-learn`, `river`, and `mapie`.
3. Chunk invariance, save/load round-trip, `write_json`, `predict()`, zero-weight
   isolation, `embargo` / `label_clock` delay, and `output_index` metadata.
4. `rp.eval.metrics` / `window_metrics` backwards compatibility plus opt-in `mae`,
   `pinball_q*`, and `d2_pinball_q*`, and `rp.eval.interval_metrics` (`coverage`,
   `mean_width`, `winkler`, `ssc`) against `scikit-learn` and `mapie`.
5. 50-seed oracle-ledger grid against `scikit-learn`, `river`, and `mapie`.
"""

from __future__ import annotations

import json
import math
from typing import Any

import mapie.metrics.regression as mapie_reg
import numpy as np
import polars as pl
import pytest
from river import metrics as river_metrics
from sklearn import metrics as sk_metrics

import ledger
from rheon import polars as rp

TIER = "mixed"


def _reg_frame(n: int = 120, *, seed: int = 0, weighted: bool = False) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    y = 0.5 + 1.4 * x0 - 0.8 * x1 + 0.6 * rng.standard_normal(n)
    cols: dict[str, Any] = {
        "t": np.arange(n, dtype=np.float64),
        "x0": x0,
        "x1": x1,
        "y": y,
    }
    if weighted:
        w = rng.uniform(0.2, 2.5, n)
        w[::17] = 0.0
        cols["w"] = w
    return pl.DataFrame(cols)


def _bin_frame(n: int = 140, *, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    logits = 0.2 + 1.2 * x0 - 0.9 * x1
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < probs).astype(np.float64)
    return pl.DataFrame(
        {
            "t": np.arange(n, dtype=np.float64),
            "x0": x0,
            "x1": x1,
            "y": y,
        }
    )


def _last_finite(series: pl.Series) -> float:
    vals = series.drop_nulls().to_numpy()
    vals = vals[np.isfinite(vals)]
    assert vals.size > 0
    return float(vals[-1])


# --- 1. Legacy parity & field selection ---------------------------------------


class TestStreamingMetricsLegacyParity:
    def test_true_matches_explicit_trio_bit_for_bit(self):
        df = _reg_frame(160, seed=7, weighted=True)
        s_bool = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=50.0,
            half_life=40.0,
            weight="w",
            emit_metrics=True,
        )
        s_list = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=50.0,
            half_life=40.0,
            weight="w",
            emit_metrics=["ic", "r2", "hit_rate"],
        )
        assert rp.spec.output_fields(s_bool) == rp.spec.output_fields(s_list)
        out_bool = rp.ModelBank([s_bool]).fit_predict(df)["m"].struct.unnest()
        out_list = rp.ModelBank([s_list]).fit_predict(df)["m"].struct.unnest()
        for col in ("ic_y", "r2_y", "hit_rate_y"):
            a = out_bool[col].to_numpy()
            b = out_list[col].to_numpy()
            np.testing.assert_array_equal(a, b)

    def test_subset_emits_only_requested_metrics(self):
        s = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            half_life=40.0,
            emit_metrics=["mae", "r2"],
        )
        fields = rp.spec.output_fields(s)
        assert "mae_y" in fields
        assert "r2_y" in fields
        assert "ic_y" not in fields
        assert "hit_rate_y" not in fields


# --- 2. Point-loss scores against oracles -------------------------------------


class TestStreamingPointScores:
    @pytest.mark.parametrize("weighted", [False, True])
    def test_mae_matches_river_and_sklearn(self, weighted: bool):
        df = _reg_frame(180, seed=11, weighted=weighted)
        kw: dict[str, Any] = {"weight": "w"} if weighted else {}
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=math.inf,
            min_weight=5.0,
            emit_metrics=["mae"],
            **kw,
        )
        out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        pred = out["pred_y"].to_numpy()
        y = df["y"].to_numpy()
        w = df["w"].to_numpy() if weighted else np.ones(len(df))
        mae_col = out["mae_y"].to_numpy()

        # Streaming row-by-row check against river.metrics.MAE (out-of-sample:
        # row i reports MAE over rows < i that had finite pred and w > 0).
        rmae = river_metrics.MAE()
        seen_y: list[float] = []
        seen_p: list[float] = []
        seen_w: list[float] = []
        for i in range(len(df)):
            if seen_y:
                assert np.isfinite(mae_col[i])
                assert abs(mae_col[i] - rmae.get()) < 1e-12
                sk = sk_metrics.mean_absolute_error(seen_y, seen_p, sample_weight=seen_w)
                assert abs(mae_col[i] - sk) < 1e-12
            else:
                assert np.isnan(mae_col[i])
            if np.isfinite(pred[i]) and w[i] > 0.0:
                rmae.update(float(y[i]), float(pred[i]), w=float(w[i]))
                seen_y.append(float(y[i]))
                seen_p.append(float(pred[i]))
                seen_w.append(float(w[i]))

    def test_pinball_default_on_quantile_and_sgd_quantile_matches_sklearn(self):
        df = _reg_frame(160, seed=13)
        for spec, tau in [
            (
                rp.spec.quantile(
                    "q",
                    targets=["y"],
                    features=["x0", "x1"],
                    quantile=0.8,
                    half_life=math.inf,
                    min_weight=5.0,
                    emit_metrics=["pinball"],
                ),
                0.8,
            ),
            (
                rp.spec.sgd(
                    "s",
                    targets=["y"],
                    features=["x0", "x1"],
                    loss="quantile",
                    quantile=0.25,
                    half_life=math.inf,
                    min_weight=5.0,
                    emit_metrics=["pinball"],
                ),
                0.25,
            ),
        ]:
            fields = rp.spec.output_fields(spec)
            assert "pinball_y" in fields
            out = rp.ModelBank([spec]).fit_predict(df)[spec["name"]].struct.unnest()
            pred = out["pred_y"].to_numpy()
            y = df["y"].to_numpy()
            pin = out["pinball_y"].to_numpy()
            mask = np.isfinite(pred)
            idx = np.flatnonzero(mask)
            # Final row reports pinball over all earlier scored rows.
            want = sk_metrics.mean_pinball_loss(y[idx[:-1]], pred[idx[:-1]], alpha=tau)
            assert abs(pin[idx[-1]] - want) < 1e-12

    def test_pinball_explicit_metric_quantiles_matches_sklearn(self):
        df = _reg_frame(160, seed=19, weighted=True)
        taus = [0.1, 0.5, 0.9]
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=math.inf,
            weight="w",
            min_weight=5.0,
            emit_metrics=["mae", "pinball"],
            metric_quantiles=taus,
        )
        idx_df = rp.spec.output_index(spec)
        pin_rows = idx_df.filter(pl.col("kind") == "pinball")
        assert pin_rows["field"].to_list() == ["pinball_q0.1_y", "pinball_q0.5_y", "pinball_q0.9_y"]
        assert pin_rows["quantile"].to_list() == taus

        out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        pred = out["pred_y"].to_numpy()
        y = df["y"].to_numpy()
        w = df["w"].to_numpy()
        valid = np.flatnonzero(np.isfinite(pred) & (w > 0.0))
        last = int(valid[-1])
        prev = valid[:-1]
        for tau, name in zip(
            taus, ["pinball_q0.1_y", "pinball_q0.5_y", "pinball_q0.9_y"], strict=True
        ):
            want = sk_metrics.mean_pinball_loss(
                y[prev], pred[prev], sample_weight=w[prev], alpha=tau
            )
            assert abs(float(out[name][last]) - want) < 1e-12
        # At tau=0.5, pinball == 0.5 * MAE.
        assert abs(float(out["pinball_q0.5_y"][last]) - 0.5 * float(out["mae_y"][last])) < 1e-13

    @pytest.mark.parametrize(
        "spec_fn",
        [
            lambda: rp.spec.sgd(
                "m",
                targets=["y"],
                features=["x0", "x1"],
                loss="logistic",
                half_life=math.inf,
                min_weight=5.0,
                emit_metrics=["log_loss", "hit_rate"],
            ),
            lambda: rp.spec.ftrl(
                "m",
                targets=["y"],
                features=["x0", "x1"],
                loss="logistic",
                half_life=math.inf,
                min_weight=5.0,
                emit_metrics=["log_loss"],
            ),
            lambda: rp.spec.glm(
                "m",
                targets=["y"],
                features=["x0", "x1"],
                family="binomial",
                half_life=math.inf,
                min_weight=5.0,
                emit_metrics=["log_loss"],
            ),
        ],
    )
    def test_log_loss_matches_sklearn_on_binary_models(self, spec_fn):
        df = _bin_frame(160, seed=23)
        spec = spec_fn()
        out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        pred = out["pred_y"].to_numpy()
        y = df["y"].to_numpy()
        ll = out["log_loss_y"].to_numpy()
        idx = np.flatnonzero(np.isfinite(pred))
        prev = idx[:-1]
        want = sk_metrics.log_loss(y[prev], pred[prev], labels=[0.0, 1.0])
        assert abs(float(ll[idx[-1]]) - want) < 1e-12

    def test_validation_refusals(self):
        # Empty list
        with pytest.raises(ValueError, match="emit_metrics"):
            rp.spec.ewridge("m", targets=["y"], features=["x0"], half_life=50.0, emit_metrics=[])
        # Duplicate entry
        with pytest.raises(ValueError, match="more than once"):
            rp.spec.ewridge(
                "m", targets=["y"], features=["x0"], half_life=50.0, emit_metrics=["mae", "mae"]
            )
        # Unknown metric name
        with pytest.raises(ValueError, match="emit_metrics"):
            rp.spec.ewridge(
                "m",
                targets=["y"],
                features=["x0"],
                half_life=50.0,
                emit_metrics=["bogus"],  # type: ignore[list-item]
            )
        # log_loss on non-binary models
        for bad in [
            lambda: rp.spec.ewridge(
                "m", targets=["y"], features=["x0"], half_life=50.0, emit_metrics=["log_loss"]
            ),
            lambda: rp.spec.ftrl(
                "m",
                targets=["y"],
                features=["x0"],
                loss="squared",
                half_life=50.0,
                emit_metrics=["log_loss"],
            ),
            lambda: rp.spec.glm(
                "m",
                targets=["y"],
                features=["x0"],
                family="poisson",
                half_life=50.0,
                emit_metrics=["log_loss"],
            ),
        ]:
            with pytest.raises(ValueError, match="log_loss"):
                bad()
        # pinball without metric_quantiles on non-quantile model
        with pytest.raises(ValueError, match="metric_quantiles"):
            rp.spec.ewridge(
                "m", targets=["y"], features=["x0"], half_life=50.0, emit_metrics=["pinball"]
            )
        # metric_quantiles without pinball
        with pytest.raises(ValueError, match="metric_quantiles"):
            rp.spec.ewridge(
                "m",
                targets=["y"],
                features=["x0"],
                half_life=50.0,
                emit_metrics=["mae"],
                metric_quantiles=[0.5],
            )
        # Invalid metric_quantiles values
        for bad_qs in ([], [0.0], [1.0], [0.5, 0.5]):
            with pytest.raises(ValueError, match="metric_quantiles"):
                rp.spec.ewridge(
                    "m",
                    targets=["y"],
                    features=["x0"],
                    half_life=50.0,
                    emit_metrics=["pinball"],
                    metric_quantiles=bad_qs,
                )
        # Interval scores without conformal
        for kind in ("coverage", "width", "winkler"):
            with pytest.raises(ValueError, match="conformal"):
                rp.spec.ewridge(
                    "m",
                    targets=["y"],
                    features=["x0"],
                    half_life=50.0,
                    emit_metrics=[kind],  # type: ignore[list-item]
                )
        # Unsupervised model refuses emit_metrics
        with pytest.raises(ValueError, match="emit_metrics"):
            rp.spec.ew_cov("m", features=["x0", "x1"], half_life=50.0, emit_metrics=["mae"])


# --- 3. Streaming interval scores against MAPIE -------------------------------


class TestStreamingIntervalScores:
    @pytest.mark.parametrize("conformal_arg", [0.9, [0.8, 0.9]])
    @pytest.mark.parametrize("symmetric", [True, False])
    def test_width_and_winkler_match_mapie(self, conformal_arg, symmetric: bool):
        df = _reg_frame(200, seed=31)
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=math.inf,
            min_weight=5.0,
            conformal=conformal_arg,
            conformal_symmetric=symmetric,
            emit_metrics=["coverage", "width", "winkler"],
        )
        fields = rp.spec.output_fields(spec)
        # No duplicate coverage field names.
        assert len(fields) == len(set(fields))

        out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        y = df["y"].to_numpy()

        levels = [conformal_arg] if isinstance(conformal_arg, float) else list(conformal_arg)
        is_multi = isinstance(conformal_arg, list)
        for lvl in levels:
            suf = f"c{lvl:g}_y" if is_multi else "y"
            lo = out[f"lo_{suf}"].to_numpy()
            hi = out[f"hi_{suf}"].to_numpy()
            width = out[f"width_{suf}"].to_numpy()
            winkler = out[f"winkler_{suf}"].to_numpy()
            cov = out[f"coverage_{suf}"].to_numpy()

            valid = np.flatnonzero(np.isfinite(lo) & np.isfinite(hi))
            assert valid.size > 50
            # At the final row `valid[-1]`, streaming scores aggregate `valid[:-1]`.
            prev = valid[:-1]
            last = int(valid[-1])
            y_pis = np.stack([lo[prev], hi[prev]], axis=1)[:, :, None]
            want_width = float(mapie_reg.regression_mean_width_score(y_pis)[0])
            want_winkler = float(mapie_reg.regression_mwi_score(y[prev], y_pis, lvl))
            want_cov = float(mapie_reg.regression_coverage_score(y[prev], y_pis)[0])
            assert abs(float(width[last]) - want_width) < 1e-12
            assert abs(float(winkler[last]) - want_winkler) < 1e-12
            assert abs(float(cov[last]) - want_cov) < 1e-12


# --- 4. Streaming invariants (chunks, save/load, zero-weight, delay, predict) -


class TestStreamingScoresInvariants:
    def test_chunk_invariance_save_load_write_json_and_predict(self, tmp_path):
        df = _reg_frame(160, seed=41, weighted=True)
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=50.0,
            half_life=30.0,
            weight="w",
            embargo=3.0,
            conformal=[0.8, 0.9],
            conformal_symmetric=False,
            emit_metrics=["ic", "mae", "pinball", "coverage", "width", "winkler"],
            metric_quantiles=[0.2, 0.8],
        )
        whole = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()

        # Random chunk sizes
        bank = rp.ModelBank([spec])
        parts = []
        pos = 0
        for sz in (1, 13, 27, 40, 19, 60):
            if pos >= len(df):
                break
            parts.append(bank.fit_predict(df.slice(pos, sz))["m"].struct.unnest())
            pos += sz
        chunked = pl.concat(parts)
        for col in whole.columns:
            if whole[col].dtype == pl.Float64:
                np.testing.assert_array_equal(chunked[col].to_numpy(), whole[col].to_numpy())

        # Save / load mid-stream + write_json
        b1 = rp.ModelBank([spec])
        head = b1.fit_predict(df.slice(0, 85))["m"].struct.unnest()
        state_path = tmp_path / "scores.msgpack"
        json_path = tmp_path / "scores.json"
        b1.save(state_path)
        b1.write_json(json_path)
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        assert parsed["kind"] == "bank"
        b2 = rp.ModelBank.load(state_path, [spec])
        tail = b2.fit_predict(df.slice(85))["m"].struct.unnest()
        resumed = pl.concat([head, tail])
        for col in ("mae_y", "pinball_q0.2_y", "width_c0.8_y", "winkler_c0.9_y"):
            np.testing.assert_array_equal(resumed[col].to_numpy(), whole[col].to_numpy())

        # predict() parity against the next row of fit_predict() with weight 0
        probe = df.slice(len(df) - 1, 1).with_columns(pl.lit(0.0).alias("w"))
        pred_out = b2.predict(probe)["m"].struct.unnest()
        fp0_out = b2.fit_predict(probe)["m"].struct.unnest()
        for col in ("mae_y", "pinball_q0.2_y", "width_c0.8_y", "winkler_c0.9_y"):
            np.testing.assert_array_equal(pred_out[col].to_numpy(), fp0_out[col].to_numpy())

    def test_zero_weight_rows_do_not_update_scores(self):
        df = _reg_frame(90, seed=51)
        w = np.ones(len(df))
        w[60:70] = 0.0
        df = df.with_columns(pl.Series("w", w))
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=50.0,
            half_life=25.0,
            weight="w",
            conformal=0.9,
            emit_metrics=["mae", "pinball", "width", "winkler"],
            metric_quantiles=[0.5],
        )
        out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        # Across zero-weight rows 60..70, pre-row score stays frozen (60..70 identical).
        for col in ("mae_y", "pinball_q0.5_y", "width_y", "winkler_y"):
            vals = out[col].to_numpy()[60:71]
            assert np.all(np.isfinite(vals))
            np.testing.assert_array_equal(vals, np.full_like(vals, vals[0]))


# --- 5. rp.eval scores & interval_metrics ------------------------------------


class TestEvalScores:
    def test_default_metrics_and_sum_fields_backwards_compatible(self):
        df = _reg_frame(100, seed=61)
        spec = rp.spec.ewridge("m", targets=["y"], features=["x0", "x1"], half_life=30.0)
        out = rp.ModelBank([spec]).fit_predict(df)
        m = rp.eval.metrics(out, spec)
        assert set(m.columns) == {
            "slot",
            "target",
            "n",
            "ic",
            "r2",
            "hit_rate",
            "mse",
            "rmse",
        }
        assert "mae" not in m.columns
        assert rp.eval.SUM_FIELDS == (
            "n",
            "w",
            "mean_y",
            "mean_pred",
            "m2_y",
            "m2_pred",
            "cov",
            "sse",
            "hits",
            "signed",
        )

    def test_eval_mae_pinball_d2_matches_sklearn_df_and_lazy(self):
        df = _reg_frame(140, seed=67, weighted=True).with_columns(
            pl.Series("g", ["a" if i % 2 == 0 else "b" for i in range(140)])
        )
        spec = rp.spec.ewridge(
            "m", targets=["y"], features=["x0", "x1"], half_life=30.0, group="g", weight="w"
        )
        out = rp.ModelBank([spec]).fit_predict(df)

        m_df = rp.eval.metrics(out, spec, group="g", mae=True, quantiles=[0.1, 0.5, 0.9])
        m_lf = rp.eval.metrics(out.lazy(), spec, group="g", mae=True, quantiles=[0.1, 0.5, 0.9])
        assert m_df.equals(m_lf)

        un = out.with_columns(out["m"].struct.unnest())
        for grp in ("a", "b"):
            sub = un.filter((pl.col("g") == grp) & pl.col("pred_y").is_not_null())
            y = sub["y"].to_numpy()
            p = sub["pred_y"].to_numpy()
            row = m_df.filter(pl.col("g") == grp).to_dicts()[0]
            assert abs(row["mae"] - sk_metrics.mean_absolute_error(y, p)) < 1e-12
            for tau in (0.1, 0.5, 0.9):
                tag = f"{tau:g}"
                want_p = sk_metrics.mean_pinball_loss(y, p, alpha=tau)
                want_d2 = sk_metrics.d2_pinball_score(y, p, alpha=tau)
                assert abs(row[f"pinball_q{tag}"] - want_p) < 1e-12
                assert abs(row[f"d2_pinball_q{tag}"] - want_d2) < 1e-12

    def test_eval_window_metrics_with_mae_and_quantiles(self):
        df = _reg_frame(120, seed=71)
        spec = rp.spec.ewridge(
            "m", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=50.0, half_life=30.0
        )
        out = rp.ModelBank([spec]).fit_predict(df)
        wm = rp.eval.window_metrics(
            out, spec, clock="t", every=40.0, mae=True, quantiles=[0.25, 0.75]
        )
        wm_lazy = rp.eval.window_metrics(
            out.lazy(), spec, clock="t", every=40.0, mae=True, quantiles=[0.25, 0.75]
        )
        assert wm.equals(wm_lazy)
        assert {
            "mae",
            "pinball_q0.25",
            "d2_pinball_q0.25",
            "pinball_q0.75",
            "d2_pinball_q0.75",
        } <= set(wm.columns)
        assert wm.height >= 2

    def test_eval_d2_pinball_degenerate_constant_y_is_null(self):
        df = pl.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [2.0] * 6})
        spec = rp.spec.ewridge("m", targets=["y"], features=["x0"], half_life=10.0, min_weight=1.0)
        out = rp.ModelBank([spec]).fit_predict(df)
        m = rp.eval.metrics(out, spec, quantiles=[0.5])
        assert m["d2_pinball_q0.5"][0] is None

    def test_eval_interval_metrics_single_and_multi_level_matches_mapie(self):
        df = _reg_frame(210, seed=79)
        for conformal_arg in (0.9, [0.8, 0.9, 0.95]):
            spec = rp.spec.ewridge(
                "m",
                targets=["y"],
                features=["x0", "x1"],
                half_life=40.0,
                min_weight=5.0,
                conformal=conformal_arg,
                conformal_symmetric=False,
            )
            out = rp.ModelBank([spec]).fit_predict(df)
            got = rp.eval.interval_metrics(out, spec, ssc_bins=3)
            got_lazy = rp.eval.interval_metrics(out.lazy(), spec, ssc_bins=3)
            assert got.equals(got_lazy)

            un = out.with_columns(out["m"].struct.unnest())
            levels = [conformal_arg] if isinstance(conformal_arg, float) else list(conformal_arg)
            is_multi = isinstance(conformal_arg, list)
            for lvl in levels:
                suf = f"c{lvl:g}_y" if is_multi else "y"
                sub = un.filter(
                    pl.col(f"lo_{suf}").is_not_null() & pl.col(f"hi_{suf}").is_not_null()
                )
                y = sub["y"].to_numpy()
                lo = sub[f"lo_{suf}"].to_numpy()
                hi = sub[f"hi_{suf}"].to_numpy()
                y_int = np.stack([lo, hi], axis=1)
                y_pis = y_int[:, :, None]

                row = got.filter(pl.col("nominal_coverage") == lvl).to_dicts()[0]
                assert row["n"] == len(y)
                assert (
                    abs(row["coverage"] - float(mapie_reg.regression_coverage_score(y, y_pis)[0]))
                    < 1e-12
                )
                assert (
                    abs(row["mean_width"] - float(mapie_reg.regression_mean_width_score(y_pis)[0]))
                    < 1e-12
                )
                assert (
                    abs(row["winkler"] - float(mapie_reg.regression_mwi_score(y, y_pis, lvl)))
                    < 1e-12
                )
                assert (
                    abs(row["ssc"] - float(mapie_reg.regression_ssc_score(y, y_pis, num_bins=3)[0]))
                    < 1e-12
                )

    def test_eval_interval_metrics_uncrosses_swapped_bounds_and_supports_group_window_weight(self):
        rng = np.random.default_rng(83)
        n = 90
        y = rng.standard_normal(n)
        lo = y - rng.uniform(0.1, 2.0, n)
        hi = y + rng.uniform(0.1, 2.0, n)
        # Artificially cross every 5th interval.
        lo_c = lo.copy()
        hi_c = hi.copy()
        lo_c[::5], hi_c[::5] = hi[::5], lo[::5]
        df = pl.DataFrame(
            {
                "g": ["a" if i < 45 else "b" for i in range(n)],
                "w": rng.uniform(0.5, 2.0, n),
                "y": y,
                "m": pl.Series(
                    "m",
                    [
                        {"pred_y": 0.0, "lo_y": float(a), "hi_y": float(b)}
                        for a, b in zip(lo_c, hi_c, strict=True)
                    ],
                ),
            }
        )
        got = rp.eval.interval_metrics(df, "m", coverage=0.9, ssc_bins=3)
        y_pis = np.stack([lo_c, hi_c], axis=1)[:, :, None]
        want_winkler = float(mapie_reg.regression_mwi_score(y, y_pis, 0.9))
        assert abs(float(got["winkler"][0]) - want_winkler) < 1e-12
        assert float(got["mean_width"][0]) > 0.0

        # Group + window_rows + weight
        gw = rp.eval.interval_metrics(df, "m", coverage=0.9, group="g", weight="w", window_rows=15)
        assert gw.height == 6  # 2 groups * 3 windows of 15
        assert "window_start" in gw.columns and "g" in gw.columns

        # Renamed `by=` keyword refusal
        with pytest.raises(TypeError, match="group"):
            rp.eval.interval_metrics(df, "m", coverage=0.9, by="g")  # type: ignore[call-arg]

        # Refusals
        with pytest.raises(ValueError, match="ssc_bins"):
            rp.eval.interval_metrics(df, "m", coverage=0.9, ssc_bins=0)
        with pytest.raises(ValueError, match="window_rows"):
            rp.eval.interval_metrics(df, "m", coverage=0.9, window_rows=0)
        with pytest.raises(ValueError, match="coverage"):
            rp.eval.interval_metrics(df, "m")


# --- 6. 50-seed oracle-ledger grid -------------------------------------------


def _scores_grid_run(seed: int) -> dict[str, Any]:
    df = _reg_frame(120, seed=seed, weighted=True)
    s_reg = rp.spec.ewridge(
        "r",
        targets=["y"],
        features=["x0", "x1"],
        half_life=math.inf,
        weight="w",
        min_weight=5.0,
        conformal=[0.8, 0.9],
        conformal_symmetric=False,
        emit_metrics=["mae", "pinball", "coverage", "width", "winkler"],
        metric_quantiles=[0.1, 0.9],
    )
    out_r = rp.ModelBank([s_reg]).fit_predict(df)
    ev_m = rp.eval.metrics(out_r, s_reg, mae=True, quantiles=[0.1, 0.9])
    ev_i = rp.eval.interval_metrics(out_r, s_reg, ssc_bins=3)

    bdf = _bin_frame(120, seed=seed + 1000)
    s_bin = rp.spec.ftrl(
        "b",
        targets=["y"],
        features=["x0", "x1"],
        loss="logistic",
        half_life=math.inf,
        min_weight=5.0,
        emit_metrics=["log_loss"],
    )
    out_b = rp.ModelBank([s_bin]).fit_predict(bdf)
    return {
        "r_un": out_r["r"].struct.unnest(),
        "b_un": out_b["b"].struct.unnest(),
        "ev_m": ev_m,
        "ev_i": ev_i,
        "df": df,
        "bdf": bdf,
    }


ledger.guard(
    "scores_oracles",
    lambda: [_scores_grid_run(seed) for seed in range(5)],
    oracles=["scikit-learn", "river", "mapie"],
    bits=52,
)


@pytest.mark.oracle_grid(reason="50-seed streaming and eval score check vs sklearn, river, mapie")
def test_fifty_seeds_scores_match_sklearn_river_and_mapie():
    mae_stream_err: list[float] = []
    pin_stream_err: list[float] = []
    ll_stream_err: list[float] = []
    winkler_stream_err: list[float] = []
    eval_d2_err: list[float] = []
    eval_ssc_err: list[float] = []
    eval_winkler_err: list[float] = []

    for seed in range(50):
        res = _scores_grid_run(seed)
        df = res["df"]
        r_un = res["r_un"]
        y = df["y"].to_numpy()
        w = df["w"].to_numpy()
        pred = r_un["pred_y"].to_numpy()

        valid = np.flatnonzero(np.isfinite(pred) & (w > 0.0))
        prev, last = valid[:-1], int(valid[-1])
        # Streaming MAE vs river & sklearn
        rmae = river_metrics.MAE()
        for idx in prev:
            rmae.update(float(y[idx]), float(pred[idx]), w=float(w[idx]))
        mae_stream_err.append(abs(float(r_un["mae_y"][last]) - rmae.get()))
        pin_stream_err.append(
            abs(
                float(r_un["pinball_q0.9_y"][last])
                - sk_metrics.mean_pinball_loss(
                    y[prev], pred[prev], sample_weight=w[prev], alpha=0.9
                )
            )
        )

        # Streaming log_loss vs sklearn
        bdf = res["bdf"]
        b_un = res["b_un"]
        bp = b_un["pred_y"].to_numpy()
        by = bdf["y"].to_numpy()
        bidx = np.flatnonzero(np.isfinite(bp))
        ll_stream_err.append(
            abs(
                float(b_un["log_loss_y"][int(bidx[-1])])
                - sk_metrics.log_loss(by[bidx[:-1]], bp[bidx[:-1]], labels=[0.0, 1.0])
            )
        )

        # Streaming Winkler (unweighted check on interval metrics + weighted on stream)
        lo9 = r_un["lo_c0.9_y"].to_numpy()
        hi9 = r_un["hi_c0.9_y"].to_numpy()
        iv = np.flatnonzero(np.isfinite(lo9) & np.isfinite(hi9))
        # Direct weighted Winkler formula check against streaming winkler
        iv_p = iv[:-1]
        iv_w = w[iv_p]
        pos_w = iv_w > 0.0
        lo_u = np.minimum(lo9[iv_p][pos_w], hi9[iv_p][pos_w])
        hi_u = np.maximum(lo9[iv_p][pos_w], hi9[iv_p][pos_w])
        yv = y[iv_p][pos_w]
        ww = iv_w[pos_w]
        pen = np.maximum(lo_u - yv, 0.0) + np.maximum(yv - hi_u, 0.0)
        want_wink_w = float(np.sum(ww * ((hi_u - lo_u) + (2.0 / 0.1) * pen)) / np.sum(ww))
        winkler_stream_err.append(abs(float(r_un["winkler_c0.9_y"][int(iv[-1])]) - want_wink_w))

        # eval D2 pinball vs sklearn (unweighted over all finite predictions)
        all_pred = np.flatnonzero(np.isfinite(pred))
        ev_m = res["ev_m"].to_dicts()[0]
        eval_d2_err.append(
            abs(
                ev_m["d2_pinball_q0.1"]
                - sk_metrics.d2_pinball_score(y[all_pred], pred[all_pred], alpha=0.1)
            )
        )

        # eval interval_metrics (SSC & Winkler) vs MAPIE
        y_pis = np.stack([lo9[iv], hi9[iv]], axis=1)[:, :, None]
        ev_i = res["ev_i"].filter(pl.col("nominal_coverage") == 0.9).to_dicts()[0]
        eval_ssc_err.append(
            abs(ev_i["ssc"] - float(mapie_reg.regression_ssc_score(y[iv], y_pis, num_bins=3)[0]))
        )
        eval_winkler_err.append(
            abs(ev_i["winkler"] - float(mapie_reg.regression_mwi_score(y[iv], y_pis, 0.9)))
        )

    for case, vals in [
        ("mae_stream_vs_river", mae_stream_err),
        ("pinball_stream_vs_sklearn", pin_stream_err),
        ("log_loss_stream_vs_sklearn", ll_stream_err),
        ("winkler_stream_exact", winkler_stream_err),
        ("d2_pinball_eval_vs_sklearn", eval_d2_err),
        ("ssc_eval_vs_mapie", eval_ssc_err),
        ("winkler_eval_vs_mapie", eval_winkler_err),
    ]:
        worst, _ = ledger.record("scores_oracles", case, vals, floor=1e-12)
        assert worst < 1e-11, (case, worst)


def test_zero_weight_rows_excluded_from_interval_metrics_and_pinball_min_samples():
    """A14 / B11: zero-weight rows do not count toward min_samples or SSC strata in
    ``interval_metrics`` or ``pinball_and_d2_frame``."""
    from rheon.polars import eval as _eval
    from rheon.polars._eval_intervals import pinball_and_d2_frame

    rng = np.random.default_rng(314)
    n = 30
    y = rng.normal(size=n)
    pred = y + 0.1 * rng.normal(size=n)
    lo = pred - rng.uniform(0.5, 2.0, size=n)
    hi = pred + rng.uniform(0.5, 2.0, size=n)
    w = np.zeros(n)
    w[-2:] = [1.0, 2.0]  # Only 2 positive-weight rows out of 30!

    df = pl.DataFrame(
        {
            "y": y,
            "w": w,
            "m": pl.DataFrame({"pred_y": pred, "lo_y": lo, "hi_y": hi}).to_struct("m"),
        }
    )

    # 1. interval_metrics: 28 zero-weight + 2 positive-weight must stay below min_samples=30.
    iv30 = rp.eval.interval_metrics(df, "m", coverage=0.9, weight="w", min_samples=30, ssc_bins=3)
    assert iv30["n"][0] == 2
    assert iv30["coverage"][0] is None and iv30["winkler"][0] is None and iv30["ssc"][0] is None

    # Even with min_samples=1, only 2 positive-weight rows < ssc_bins=3 -> ssc is None,
    # while coverage/mean_width/winkler match the positive-weight subset to the bit.
    iv1 = rp.eval.interval_metrics(df, "m", coverage=0.9, weight="w", min_samples=1, ssc_bins=3)
    iv_pos = rp.eval.interval_metrics(
        df.filter(pl.col("w") > 0), "m", coverage=0.9, weight="w", min_samples=1, ssc_bins=3
    )
    assert iv1.equals(iv_pos)
    assert iv1["n"][0] == 2 and iv1["ssc"][0] is None and iv1["coverage"][0] is not None

    # 2. pinball_and_d2_frame: 28 zero-weight + 2 positive-weight withheld at min_samples=30.
    slots = [("pred_y", "y", pl.col("y"))]
    q30 = pinball_and_d2_frame(
        df,
        "m",
        slots,
        ["w"],
        ["slot", "target"],
        [0.5],
        30,
        weight="w",
        scored_fn=_eval._scored,
        weighed_fn=_eval._weighed,
    )
    assert q30["pinball_q0.5"][0] is None and q30["d2_pinball_q0.5"][0] is None

    # 29 zero-weight + 1 positive-weight: at min_samples=1, pinball computes, d2_pinball is None.
    w1 = np.zeros(n)
    w1[-1] = 1.5
    df1 = df.with_columns(pl.Series("w", w1))
    q1 = pinball_and_d2_frame(
        df1,
        "m",
        slots,
        ["w"],
        ["slot", "target"],
        [0.5],
        1,
        weight="w",
        scored_fn=_eval._scored,
        weighed_fn=_eval._weighed,
    )
    assert q1["pinball_q0.5"][0] is not None and q1["d2_pinball_q0.5"][0] is None
