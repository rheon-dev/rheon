"""Tests for conformal prediction variants (`conformal=[...]`,
`conformal_symmetric=False`, standalone `emit_conformal_p` with row or clock
windows) and top-level `max_state_bytes` budgets.

Verifies:
1. Asymmetric conformal (`conformal_symmetric=False`): separate signed lower
   and upper tail trackers, negative multiplier support on skewed residuals,
   narrower intervals than symmetric on skewed noise, `conformal_rate="auto"`,
   `embargo`, and `label_clock`.
2. Multi-level conformal (`conformal=[0.8, 0.9, 0.95]`): `lo_c<lvl>_<slot>`,
   `hi_c<lvl>_<slot>`, `coverage_c<lvl>_<slot>`, `output_index` metadata,
   non-crossing guarantee on every row, and validation refusals.
3. Standalone `emit_conformal_p=True` (without `conformal=`): exact unsmoothed
   p-values matching `crepes` (`smoothing=False`), row and clock-span windows
   (numeric & temporal `Duration`), Rule 9 zero-weight isolation, `embargo` /
   `label_clock` release, chunk invariance, save/load, and `predict()` parity.
4. Top-level `max_state_bytes`: static refusal, runtime half-cap warning and
   atomic cap refusal on `conformal_p_window_size`, ceiling enforcement, and
   backwards compatibility across existing models.
5. 50-seed oracle-ledger grid against `crepes` and `mapie`.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import warnings
from typing import Any

import crepes
import mapie.conformity_scores as mapie_cs
import mapie.regression as mapie_reg
import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

import ledger
import rheon
from rheon import polars as rp

TIER = "mixed"


def _skewed_frame(n: int = 800, *, seed: int = 0) -> pl.DataFrame:
    """Linear stream with right-skewed shifted-exponential noise (mean 0)."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    # Shifted Exp(1) - 1 has mean 0, strong right skew, lower bound -1.
    noise = rng.exponential(scale=1.0, size=n) - 1.0
    y = 0.5 + 1.5 * x0 - 0.7 * x1 + noise
    return pl.DataFrame(
        {
            "t": np.arange(n, dtype=np.float64),
            "x0": x0,
            "x1": x1,
            "y": y,
        }
    )


def _gauss_frame(n: int = 160, *, seed: int = 0, weighted: bool = False) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    y = 0.3 + 1.2 * x0 - 0.5 * x1 + 0.7 * rng.standard_normal(n)
    cols: dict[str, Any] = {
        "t": np.arange(n, dtype=np.float64),
        "x0": x0,
        "x1": x1,
        "y": y,
    }
    if weighted:
        w = rng.uniform(0.3, 2.0, n)
        w[::19] = 0.0
        cols["w"] = w
    return pl.DataFrame(cols)


# --- 1. Asymmetric conformal --------------------------------------------------


class TestAsymmetricConformal:
    def test_skewed_noise_adapts_asymmetrically_and_narrows_width(self):
        # Strongly right-skewed shifted noise: E[e] = 0, lower tail bounded at -1,
        # upper tail long. Asymmetric bounds stay tight on the left and widen on the right.
        df = _skewed_frame(4000, seed=5)
        s_sym = rp.spec.ewridge(
            "sym",
            targets=["y"],
            features=["x0", "x1"],
            half_life=200.0,
            min_weight=10.0,
            conformal=0.9,
            conformal_symmetric=True,
        )
        s_asym = rp.spec.ewridge(
            "asym",
            targets=["y"],
            features=["x0", "x1"],
            half_life=200.0,
            min_weight=10.0,
            conformal=0.9,
            conformal_symmetric=False,
        )
        out = rp.ModelBank([s_sym, s_asym]).fit_predict(df)
        y = df["y"].to_numpy()
        sym = out["sym"].struct.unnest()
        asym = out["asym"].struct.unnest()

        sl = slice(1000, None)
        sym_lo, sym_hi = sym["lo_y"].to_numpy()[sl], sym["hi_y"].to_numpy()[sl]
        asym_lo, asym_hi = asym["lo_y"].to_numpy()[sl], asym["hi_y"].to_numpy()[sl]
        asym_pred = asym["pred_y"].to_numpy()[sl]

        # Both achieve ~90% coverage.
        cov_sym = np.mean((y[sl] >= sym_lo) & (y[sl] <= sym_hi))
        cov_asym = np.mean((y[sl] >= asym_lo) & (y[sl] <= asym_hi))
        assert abs(cov_sym - 0.9) < 0.025
        assert abs(cov_asym - 0.9) < 0.025

        # Upper offset is significantly larger than lower offset under right skew.
        lo_off = np.mean(asym_pred - asym_lo)
        hi_off = np.mean(asym_hi - asym_pred)
        assert hi_off > 1.5 * lo_off

        # Equal-tail balance: asymmetric puts ~5% miss on each tail, whereas symmetric
        # puts nearly 0% miss on the bounded lower tail and ~10% on the upper tail.
        asym_lo_miss = float(np.mean(y[sl] < asym_lo))
        asym_hi_miss = float(np.mean(y[sl] > asym_hi))
        sym_lo_miss = float(np.mean(y[sl] < sym_lo))
        assert abs(asym_lo_miss - 0.05) < 0.02
        assert abs(asym_hi_miss - 0.05) < 0.02
        assert sym_lo_miss < 0.01
        assert np.mean(asym_lo) > np.mean(sym_lo) + 0.2

    def test_negative_multiplier_supported_on_biased_residuals(self):
        # Constant positive bias on a no-intercept regression with positive y:
        # resid = y - pred is strictly positive, so lo = pred - m_lo * sigma > pred (m_lo < 0).
        rng = np.random.default_rng(7)
        n = 1200
        x0 = rng.standard_normal(n) * 0.1
        y = 5.0 + 0.1 * rng.standard_normal(n)
        df = pl.DataFrame({"x0": x0, "y": y})
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            fit_intercept=False,
            half_life=100.0,
            min_weight=5.0,
            conformal=0.9,
            conformal_symmetric=False,
            conformal_rate=0.2,
        )
        out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        lo = out["lo_y"].to_numpy()[600:]
        pred = out["pred_y"].to_numpy()[600:]
        # With no 0-clamp on m_lo, lo rises above pred when all residuals are positive.
        assert np.mean(lo > pred) > 0.8

    def test_asymmetric_works_with_auto_rate_embargo_and_label_clock(self):
        df = _skewed_frame(200, seed=11).with_columns(
            (pl.col("t") + pl.Series([2.0, 5.0, 1.0, 4.0] * 50)).alias("avail")
        )
        s_emb = rp.spec.ewridge(
            "emb",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=50.0,
            half_life=50.0,
            embargo=4.0,
            conformal=0.9,
            conformal_symmetric=False,
            conformal_rate="auto",
        )
        s_lc = rp.spec.ewridge(
            "lc",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=50.0,
            half_life=50.0,
            label_clock="avail",
            conformal=0.9,
            conformal_symmetric=False,
            conformal_rate="auto",
        )
        out = rp.ModelBank([s_emb, s_lc]).fit_predict(df)
        for name in ("emb", "lc"):
            un = out[name].struct.unnest()
            assert un["lo_y"].drop_nulls().len() > 100
            assert un["hi_y"].drop_nulls().len() > 100

    def test_asymmetric_refused_without_conformal(self):
        with pytest.raises(ValueError, match="conformal_symmetric"):
            rp.spec.ewridge(
                "m",
                targets=["y"],
                features=["x0"],
                half_life=50.0,
                conformal_symmetric=False,
            )


# --- 2. Multi-level conformal -------------------------------------------------


class TestMultiLevelConformal:
    @pytest.mark.parametrize("symmetric", [True, False])
    def test_multi_level_names_metadata_and_non_crossing(self, symmetric: bool):
        # Aggressive rate + mid-stream regime shift to stress-test non-crossing.
        rng = np.random.default_rng(17)
        n = 500
        x0 = rng.standard_normal(n)
        noise = rng.standard_normal(n) * np.where(np.arange(n) < 250, 0.5, 4.0)
        df = pl.DataFrame({"x0": x0, "y": 1.0 + 2.0 * x0 + noise})

        levels = [0.8, 0.9, 0.95]
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            half_life=50.0,
            min_weight=5.0,
            conformal=levels,
            conformal_symmetric=symmetric,
            conformal_rate=0.5,
        )
        idx = rp.spec.output_index(spec)
        lo_rows = idx.filter(pl.col("kind") == "lo")
        assert lo_rows["field"].to_list() == ["lo_c0.8_y", "lo_c0.9_y", "lo_c0.95_y"]
        assert lo_rows["coverage"].to_list() == levels

        out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        lo8, lo9, lo95 = (out[f"lo_c{c:g}_y"].to_numpy() for c in levels)
        hi8, hi9, hi95 = (out[f"hi_c{c:g}_y"].to_numpy() for c in levels)
        ok = np.isfinite(lo8) & np.isfinite(lo9) & np.isfinite(lo95)
        assert ok.sum() > 400
        # Nested non-crossing guarantee on EVERY finite row.
        assert np.all(lo8[ok] >= lo9[ok])
        assert np.all(lo9[ok] >= lo95[ok])
        assert np.all(hi8[ok] <= hi9[ok])
        assert np.all(hi9[ok] <= hi95[ok])

    def test_single_element_list_matches_scalar_values(self):
        df = _gauss_frame(120, seed=21)
        s_scalar = rp.spec.ewridge(
            "s", targets=["y"], features=["x0", "x1"], half_life=40.0, conformal=0.9
        )
        s_list = rp.spec.ewridge(
            "l", targets=["y"], features=["x0", "x1"], half_life=40.0, conformal=[0.9]
        )
        assert "lo_y" in rp.spec.output_fields(s_scalar)
        assert "lo_c0.9_y" in rp.spec.output_fields(s_list)
        out = rp.ModelBank([s_scalar, s_list]).fit_predict(df)
        a = out["s"].struct.unnest()
        b = out["l"].struct.unnest()
        for src, dst in [
            ("lo_y", "lo_c0.9_y"),
            ("hi_y", "hi_c0.9_y"),
            ("coverage_y", "coverage_c0.9_y"),
        ]:
            np.testing.assert_array_equal(a[src].to_numpy(), b[dst].to_numpy())

    def test_multi_level_validation_refusals(self):
        for bad in ([], [0.9, 0.8], [0.9, 0.9], [0.0, 0.9], [0.8, 1.0], [-0.1, 0.5]):
            with pytest.raises(ValueError, match="conformal"):
                rp.spec.ewridge("m", targets=["y"], features=["x0"], half_life=50.0, conformal=bad)


# --- 3. Standalone conformal p-value ------------------------------------------


class TestStandaloneConformalP:
    def test_standalone_without_conformal_matches_crepes_exact(self):
        df = _gauss_frame(150, seed=29)
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=math.inf,
            min_weight=5.0,
            emit_sigma=True,
            emit_conformal_p=True,
            conformal_p_window_rows=500,
        )
        fields = rp.spec.output_fields(spec)
        assert "conformal_p_y" in fields
        assert "lo_y" not in fields

        out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        pred = out["pred_y"].to_numpy()
        sig = out["sigma_y"].to_numpy()
        y = df["y"].to_numpy()
        p_got = out["conformal_p_y"].to_numpy()

        # Rows where both pred and sigma are finite (sigma > 0).
        valid = np.flatnonzero(np.isfinite(pred) & np.isfinite(sig) & (sig > 0.0))
        assert valid.size > 80
        # Very first scored row has an empty calibration ring -> null/NaN.
        assert np.isnan(p_got[valid[0]])

        # crepes fitted on the 1st scored row's residual/sigma, then online over the rest.
        first = valid[0]
        rest = valid[1:]
        cr = crepes.ConformalRegressor()
        cr.fit(residuals=np.array([y[first] - pred[first]]), sigmas=np.array([sig[first]]))
        p_crepes = cr.predict_p_online(
            y_hat=pred[rest],
            y=y[rest],
            sigmas=sig[rest],
            smoothing=False,
            warm_start=True,
        )
        np.testing.assert_allclose(p_got[rest], p_crepes, rtol=0.0, atol=1e-13)

    def test_row_window_evicts_old_entries(self):
        df = _gauss_frame(120, seed=37)
        win = 15
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=math.inf,
            min_weight=5.0,
            emit_sigma=True,
            emit_conformal_p=True,
            conformal_p_window_rows=win,
        )
        out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        pred = out["pred_y"].to_numpy()
        sig = out["sigma_y"].to_numpy()
        y = df["y"].to_numpy()
        p_got = out["conformal_p_y"].to_numpy()

        ring: list[float] = []
        for i in range(len(df)):
            if np.isfinite(pred[i]) and np.isfinite(sig[i]) and sig[i] > 0.0:
                a_t = abs(float(y[i] - pred[i])) / float(sig[i])
                if ring:
                    ge = sum(1.0 for a in ring if a >= a_t)
                    want = (ge + 1.0) / (len(ring) + 1.0)
                    assert abs(float(p_got[i]) - want) < 1e-13
                else:
                    assert np.isnan(p_got[i])
                ring.append(a_t)
                if len(ring) > win:
                    ring.pop(0)

    def test_clock_window_numeric_and_temporal_expiry_and_predict(self):
        # Temporal Datetime clock with timedelta window_size
        n = 100
        rng = np.random.default_rng(43)
        base = dt.datetime(2026, 1, 1, 0, 0, 0)
        ts = [base + dt.timedelta(seconds=2 * i) for i in range(n)]
        x0 = rng.standard_normal(n)
        y = 1.0 + 1.5 * x0 + 0.5 * rng.standard_normal(n)
        df = pl.DataFrame({"ts": ts, "x0": x0, "y": y})

        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="ts",
            gap_cap=dt.timedelta(minutes=10),
            half_life=math.inf,
            min_weight=5.0,
            emit_sigma=True,
            emit_conformal_p=True,
            conformal_p_window_size=dt.timedelta(seconds=20),
        )
        bank = rp.ModelBank([spec])
        out = bank.fit_predict(df)["m"].struct.unnest()
        assert out["conformal_p_y"].drop_nulls().len() > 70

        # A predict() row far in the future expires every entry in the 20s window -> null.
        future = pl.DataFrame({"ts": [ts[-1] + dt.timedelta(seconds=30)], "x0": [0.5], "y": [1.5]})
        pf = bank.predict(future)["m"].struct.unnest()
        assert pf["conformal_p_y"][0] is None

        # A predict() row within the 20s window scores against non-expired entries.
        near = pl.DataFrame({"ts": [ts[-1] + dt.timedelta(seconds=2)], "x0": [0.5], "y": [1.5]})
        pn = bank.predict(near)["m"].struct.unnest()
        assert pn["conformal_p_y"][0] is not None and 0.0 < pn["conformal_p_y"][0] <= 1.0

    def test_zero_weight_rule_9_and_embargo(self):
        df = _gauss_frame(100, seed=47)
        w = np.ones(len(df))
        w[50:55] = 0.0
        df = df.with_columns(pl.Series("w", w))
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=50.0,
            half_life=math.inf,
            weight="w",
            min_weight=5.0,
            emit_conformal_p=True,
            conformal_p_window_rows=30,
        )
        # Fit rows 0..50, check predict() vs fit_predict() on zero-weight rows 50..55.
        b_pred = rp.ModelBank([spec])
        b_pred.fit_predict(df.slice(0, 50))
        pred_p = b_pred.predict(df.slice(50, 5))["m"].struct.unnest()["conformal_p_y"].to_numpy()

        b_fp = rp.ModelBank([spec])
        b_fp.fit_predict(df.slice(0, 50))
        fp_p = b_fp.fit_predict(df.slice(50, 5))["m"].struct.unnest()["conformal_p_y"].to_numpy()
        np.testing.assert_array_equal(fp_p, pred_p)

        # And state after zero-weight rows continues identically to skipping them on a row clock.
        spec_emb = rp.spec.ewridge(
            "e",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=50.0,
            half_life=30.0,
            embargo=5.0,
            emit_conformal_p=True,
            conformal_p_window_size=25.0,
        )
        whole = rp.ModelBank([spec_emb]).fit_predict(df)["e"].struct.unnest()
        b_ch = rp.ModelBank([spec_emb])
        parts = [
            b_ch.fit_predict(df.slice(0, 33))["e"].struct.unnest(),
            b_ch.fit_predict(df.slice(33, 34))["e"].struct.unnest(),
            b_ch.fit_predict(df.slice(67))["e"].struct.unnest(),
        ]
        np.testing.assert_array_equal(
            pl.concat(parts)["conformal_p_y"].to_numpy(), whole["conformal_p_y"].to_numpy()
        )

    def test_conformal_p_save_load_and_refusals(self, tmp_path):
        df = _gauss_frame(110, seed=53, weighted=True)
        spec = rp.spec.rls(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=50.0,
            half_life=40.0,
            weight="w",
            embargo=3.0,
            emit_conformal_p=True,
            conformal_p_window_size=20.0,
        )
        whole = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        b1 = rp.ModelBank([spec])
        head = b1.fit_predict(df.slice(0, 55))["m"].struct.unnest()
        p = tmp_path / "cp.msgpack"
        jp = tmp_path / "cp.json"
        b1.save(p)
        b1.write_json(jp)
        assert json.loads(jp.read_text(encoding="utf-8"))["kind"] == "bank"
        tail = rp.ModelBank.load(p, [spec]).fit_predict(df.slice(55))["m"].struct.unnest()
        np.testing.assert_array_equal(
            pl.concat([head, tail])["conformal_p_y"].to_numpy(),
            whole["conformal_p_y"].to_numpy(),
        )

        # Validation refusals
        with pytest.raises(ValueError, match="conformal_p_window_rows"):
            rp.spec.ewridge(
                "m",
                targets=["y"],
                features=["x0"],
                half_life=50.0,
                conformal_p_window_rows=50,
            )
        with pytest.raises(ValueError, match="conformal_p_window_size"):
            rp.spec.ewridge(
                "m",
                targets=["y"],
                features=["x0"],
                clock="t",
                gap_cap=50.0,
                half_life=50.0,
                conformal_p_window_size=10.0,
            )
        with pytest.raises(ValueError, match="two windows"):
            rp.spec.ewridge(
                "m",
                targets=["y"],
                features=["x0"],
                clock="t",
                gap_cap=50.0,
                half_life=50.0,
                emit_conformal_p=True,
                conformal_p_window_rows=50,
                conformal_p_window_size=10.0,
            )
        with pytest.raises(ValueError, match="clock"):
            rp.spec.ewridge(
                "m",
                targets=["y"],
                features=["x0"],
                half_life=50.0,
                emit_conformal_p=True,
                conformal_p_window_size=dt.timedelta(seconds=10),
            )
        with pytest.raises(ValueError, match="emit_conformal_p"):
            rp.spec.ew_cov("m", features=["x0", "x1"], half_life=50.0, emit_conformal_p=True)


# --- 4. Top-level max_state_bytes ---------------------------------------------


class TestTopLevelMaxStateBytes:
    def test_static_refusal_and_runtime_warn_and_atomic_cap(self):
        # Ceiling > 64 MiB refused
        with pytest.raises(ValueError, match="max_state_bytes"):
            rp.spec.ewridge(
                "m",
                targets=["y"],
                features=["x0"],
                half_life=50.0,
                max_state_bytes=64 * 1024 * 1024 + 1,
            )
        # Floor <= 0 refused
        for bad in (0, -10):
            with pytest.raises(ValueError, match="max_state_bytes"):
                rp.spec.ewridge(
                    "m", targets=["y"], features=["x0"], half_life=50.0, max_state_bytes=bad
                )
        # Static refusal when row window exceeds max_state_bytes
        with pytest.raises(ValueError, match="max_state_bytes"):
            rp.spec.ewridge(
                "m",
                targets=["y"],
                features=["x0"],
                half_life=50.0,
                emit_conformal_p=True,
                conformal_p_window_rows=5000,
                max_state_bytes=4096,
            )

        # Runtime half-cap RheonWarning, then atomic refusal on clock window overflow.
        n = 250
        df = _gauss_frame(n, seed=61)
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=500.0,
            half_life=math.inf,
            min_weight=3.0,
            emit_conformal_p=True,
            conformal_p_window_size=500.0,
            max_state_bytes=2048,
        )
        bank = rp.ModelBank([spec])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Feed enough rows to cross half-cap without hitting full cap.
            bank.fit_predict(df.slice(0, 45))
        assert any(issubclass(w.category, rheon.RheonWarning) for w in caught)

        # Feed remaining rows -> exceeds 2048 B cap and refuses atomically.
        with pytest.raises(ValueError, match="max_state_bytes"):
            bank.fit_predict(df.slice(45))

    def test_top_level_max_state_bytes_compatible_with_existing_models(self):
        specs = [
            rp.spec.hawkes(
                "h",
                targets=["y"],
                kernel_decays=[1.0],
                clock="t",
                gap_cap=50.0,
                half_life=math.inf,
                window_size=20.0,
                max_state_bytes=100_000,
            ),
            rp.spec.glm(
                "g",
                targets=["y"],
                features=["x0"],
                clock="t",
                gap_cap=50.0,
                half_life=math.inf,
                refit_span=20.0,
                max_state_bytes=100_000,
            ),
            rp.spec.corrchange(
                "c",
                features=["x0", "x1"],
                clock="t",
                gap_cap=50.0,
                window_size=20.0,
                max_state_bytes=100_000,
            ),
            rp.spec.spectral(
                "sp",
                targets=["y"],
                periods=[8.0],
                clock="t",
                gap_cap=50.0,
                half_life=50.0,
                window_size=20.0,
                max_state_bytes=100_000,
            ),
        ]
        bank = rp.ModelBank(specs)
        assert bank.specs == specs


# --- 5. 50-seed oracle-ledger grid -------------------------------------------


def _crepes_grid_run(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = _gauss_frame(120, seed=seed)
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=math.inf,
        min_weight=5.0,
        emit_sigma=True,
        emit_conformal_p=True,
        conformal_p_window_rows=500,
    )
    out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
    return (
        df["y"].to_numpy(),
        out["pred_y"].to_numpy(),
        out["sigma_y"].to_numpy(),
        out["conformal_p_y"].to_numpy(),
    )


def _asym_multi_grid_run(seed: int) -> dict[str, Any]:
    df = _skewed_frame(1200, seed=seed)
    levels = [0.8, 0.9, 0.95]
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=math.inf,
        min_weight=10.0,
        conformal=levels,
        conformal_symmetric=False,
        conformal_rate=0.05,
    )
    out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
    return {"df": df, "out": out}


ledger.guard(
    "conformal_variants_oracles",
    lambda: (
        [_crepes_grid_run(seed) for seed in range(5)],
        [_asym_multi_grid_run(seed)["out"].select("lo_c0.9_y", "hi_c0.9_y") for seed in range(3)],
    ),
    oracles=["crepes", "mapie"],
)


@pytest.mark.oracle_grid(reason="50-seed conformal_p vs crepes and asymmetric multi-level vs mapie")
def test_fifty_seeds_conformal_variants_vs_crepes_and_mapie():
    crepes_max_err: list[float] = []
    asym_cov_err_90: list[float] = []
    mapie_width_ratio_90: list[float] = []

    levels = [0.8, 0.9, 0.95]
    for seed in range(50):
        # 1. Exact conformal_p vs crepes (smoothing=False)
        y, pred, sig, p_got = _crepes_grid_run(seed)
        valid = np.flatnonzero(np.isfinite(pred) & np.isfinite(sig) & (sig > 0.0))
        first, rest = valid[0], valid[1:]
        cr = crepes.ConformalRegressor()
        cr.fit(residuals=np.array([y[first] - pred[first]]), sigmas=np.array([sig[first]]))
        p_ref = cr.predict_p_online(
            y_hat=pred[rest], y=y[rest], sigmas=sig[rest], smoothing=False, warm_start=True
        )
        crepes_max_err.append(float(np.max(np.abs(p_got[rest] - p_ref))))

        # 2. Asymmetric multi-level conformal coverage & width vs MAPIE SplitConformalRegressor
        res = _asym_multi_grid_run(seed)
        df = res["df"]
        out = res["out"]
        x = df.select("x0", "x1").to_numpy()
        yv = df["y"].to_numpy()

        # Non-crossing check on all 50 seeds
        lo8, lo9, lo95 = (out[f"lo_c{c:g}_y"].to_numpy() for c in levels)
        hi8, hi9, hi95 = (out[f"hi_c{c:g}_y"].to_numpy() for c in levels)
        sl = slice(400, None)
        assert np.all(lo8[sl] >= lo9[sl]) and np.all(lo9[sl] >= lo95[sl])
        assert np.all(hi8[sl] <= hi9[sl]) and np.all(hi9[sl] <= hi95[sl])

        cov9 = float(np.mean((yv[sl] >= lo9[sl]) & (yv[sl] <= hi9[sl])))
        asym_cov_err_90.append(abs(cov9 - 0.9))

        # Fit MAPIE SplitConformalRegressor(sym=False) on train=0..400, cal=400..800, test=800..1200
        est = LinearRegression().fit(x[:400], yv[:400])
        scr = mapie_reg.SplitConformalRegressor(
            estimator=est,
            confidence_level=levels,
            conformity_score=mapie_cs.AbsoluteConformityScore(sym=False),
            prefit=True,
        )
        scr.conformalize(x[400:800], yv[400:800])
        _, pis = scr.predict_interval(x[800:])
        # Level 0.9 is index 1
        mapie_w = float(np.mean(pis[:, 1, 1] - pis[:, 0, 1]))
        rheon_w = float(np.mean(hi9[800:] - lo9[800:]))
        mapie_width_ratio_90.append(rheon_w / mapie_w)

    worst_p, _ = ledger.record(
        "conformal_variants_oracles", "conformal_p_vs_crepes", crepes_max_err, floor=1e-12
    )
    assert worst_p < 1e-11, worst_p
    worst_cov, _ = ledger.record(
        "conformal_variants_oracles", "asym_cov_err_at_0p9", asym_cov_err_90
    )
    assert worst_cov < 0.05, worst_cov
    worst_ratio, _ = ledger.record(
        "conformal_variants_oracles", "asym_width_ratio_vs_mapie", mapie_width_ratio_90, ideal=1.0
    )
    assert 0.7 < worst_ratio < 1.3, worst_ratio


def test_default_conformal_p_window_scales_with_row_clock_half_life():
    """On a row clock with finite half_life h, default conformal_p_window_rows
    scales to ceil(10 * h) (floored at 10, capped by max_state_bytes); under
    infinite half_life or an explicit clock it stays 1,000."""
    import json

    from rheon import _rheon as _native
    from rheon import _spec

    def resolved_win(**kw):
        spec = rp.spec.ewridge("m", targets=["y"], features=["x0"], emit_conformal_p=True, **kw)
        return json.loads(_native.resolved_defaults(_spec._json(spec)))["stream"][
            "conformal_p_window_rows"
        ]

    assert resolved_win(half_life=25.0) == 250
    assert resolved_win(half_life=0.5) == 10
    assert resolved_win(half_life=100_000.0) == (262_144 - 64) // 24
    assert resolved_win(half_life=math.inf) == 1000
    assert resolved_win(half_life=25.0, clock="t", gap_cap=10.0) == 1000
    assert resolved_win(half_life=25.0, conformal_p_window_rows=77) == 77

    # End-to-end: row-clock half_life=2.0 matches explicit conformal_p_window_rows=20 to the bit.
    df = _gauss_frame(80, seed=71).drop("t")
    s_auto = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=2.0,
        min_weight=3.0,
        emit_conformal_p=True,
    )
    s_explicit = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=2.0,
        min_weight=3.0,
        emit_conformal_p=True,
        conformal_p_window_rows=20,
    )
    a = rp.ModelBank([s_auto]).fit_predict(df)["m"].struct.field("conformal_p_y")
    b = rp.ModelBank([s_explicit]).fit_predict(df)["m"].struct.field("conformal_p_y")
    assert a.equals(b)
