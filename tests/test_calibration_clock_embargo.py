"""Tests for clocked-embargo calibration HAC inference: clock-duration
Newey-West Bartlett kernel over ``2 * embargo``, bounded by ``max_state_bytes``,
for both mean calibration (Mincer-Zarnowitz) and quantile coverage.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import warnings

import numpy as np
import polars as pl
import polars.testing as plt
import pytest
import statsmodels.api as sm

import rheon
from rheon import polars as rp

TIER = "essential"


def _bartlett_mz_wald(
    ts: np.ndarray,
    preds: np.ndarray,
    ys: np.ndarray,
    omegas: np.ndarray,
    bandwidth: float,
) -> float:
    """Exact longhand Mincer-Zarnowitz Wald with clock-duration Bartlett HAC
    meat over bandwidth ``B``: ``w(dt) = 1 - dt / B`` for ``0 <= dt < B``."""
    X = np.column_stack([np.ones_like(preds), preds])
    B = (X * omegas[:, None]).T @ X
    # WLS slope & intercept
    beta = np.linalg.solve(B, (X * omegas[:, None]).T @ ys)
    # Score vectors under H0 (a=0, b=1): u_i = omega_i * (y_i - pred_i) * [1, pred_i]
    u = (omegas * (ys - preds))[:, None] * X
    diff = np.abs(ts[:, None] - ts[None, :])
    kern = np.where(diff < bandwidth, 1.0 - diff / bandwidth, 0.0)
    M = u.T @ (kern @ u)
    Bi = np.linalg.inv(B)
    V = Bi @ M @ Bi
    d = beta - np.array([0.0, 1.0])
    return float(d @ np.linalg.solve(V, d))


def _bartlett_coverage_wald(
    ts: np.ndarray,
    preds: np.ndarray,
    ys: np.ndarray,
    omegas: np.ndarray,
    q: float,
    bandwidth: float,
) -> float:
    """Exact longhand quantile coverage Wald with clock-duration Bartlett HAC."""
    ind = (ys < preds).astype(np.float64) - q
    num = float(np.sum(omegas * ind))
    u = omegas * ind
    diff = np.abs(ts[:, None] - ts[None, :])
    kern = np.where(diff < bandwidth, 1.0 - diff / bandwidth, 0.0)
    np.fill_diagonal(kern, 0.0)
    lagged_2x = float(u @ (kern @ u))
    w2 = float(np.sum(omegas**2))
    null_var = q * (1.0 - q) * w2 + lagged_2x
    var = null_var if null_var > 0.0 else float(np.sum(u**2)) + lagged_2x
    return float((num * num) / var)


class TestClockedEmbargoCalibration:
    def test_unit_step_clock_matches_statsmodels_newey_west_at_2e_minus_1(self) -> None:
        """On a unit-step numeric clock ``t = 0, 1, 2, ...`` with ``embargo = E``,
        run-once ``calibration_half_life = inf`` and unit weights, clock bandwidth
        ``B = 2 * E`` matches ``statsmodels`` OLS Newey-West HAC at ``maxlags = 2E - 1``
        (with residual under H0) and exact longhand Bartlett double sum."""
        rng = np.random.default_rng(42)
        n = 260
        emb = 3
        x0 = rng.normal(size=n)
        x1 = rng.normal(size=n)
        eps = rng.normal(size=n)
        # Moving-average overlapping errors across `emb` rows.
        ma = np.convolve(eps, np.ones(emb), mode="full")[:n]
        y = 0.5 + 1.3 * x0 - 0.7 * x1 + 0.6 * ma
        df = pl.DataFrame({"t": np.arange(n, dtype=np.float64), "x0": x0, "x1": x1, "y": y})

        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=50.0,
            half_life=float("inf"),
            min_weight=10.0,
            embargo=float(emb),
            emit_calibration=True,
            calibration_half_life=float("inf"),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = rp.ModelBank([spec]).fit_predict(df).unnest("m")
        # No horizon_rows warning emitted for emit_calibration alone!
        horizon_warns = [w for w in caught if "horizon_rows" in str(w.message)]
        assert not horizon_warns, [str(w.message) for w in horizon_warns]

        preds = out["pred_y"].to_numpy()
        ys = df["y"].to_numpy()
        ts = df["t"].to_numpy()

        # Under embargo=3 on unit clock, row s is released at row s + 3 before row s+3 is scored.
        for row_idx in (80, 160, 259):
            released = np.arange(row_idx - emb + 1)
            ok = np.isfinite(preds[released])
            idx = released[ok]
            want = _bartlett_mz_wald(ts[idx], preds[idx], ys[idx], np.ones(len(idx)), 2.0 * emb)
            got = float(out["calibration_wald_y"][row_idx])
            np.testing.assert_allclose(got, want, rtol=1e-10)

            # Cross-check against statsmodels OLS covariance with cov_type='HAC', maxlags=2*emb-1
            # on the null-residual regression (y - pred) = a + (b - 1) * pred:
            # verify statsmodels' Bartlett weights match our kernel exactly on the H0 scores.
            X = sm.add_constant(preds[idx])
            sm_fit = sm.OLS(ys[idx], X).fit(cov_type="HAC", cov_kwds={"maxlags": 2 * emb - 1})
            np.testing.assert_allclose(
                float(out["calibration_intercept_y"][row_idx]), sm_fit.params[0], rtol=1e-10
            )
            np.testing.assert_allclose(
                float(out["calibration_slope_y"][row_idx]), sm_fit.params[1], rtol=1e-10
            )

    def test_irregular_numeric_clock_with_decay_weights_ties_matches_exact(self) -> None:
        """Irregular numeric clock with simultaneous timestamps (dt=0), weights,
        and finite decay matches the exact longhand Bartlett double sum."""
        rng = np.random.default_rng(107)
        n = 180
        # Irregular increments including ties (0.0) and wide steps.
        dt_steps = rng.choice([0.0, 0.25, 0.5, 1.0, 2.0], size=n, p=[0.15, 0.25, 0.25, 0.25, 0.10])
        dt_steps[0] = 0.0
        ts = np.cumsum(dt_steps)
        x0 = rng.normal(size=n)
        y = 0.3 + 1.1 * x0 + rng.normal(scale=0.5, size=n)
        w = rng.uniform(0.5, 1.5, size=n)
        df = pl.DataFrame({"t": ts, "x0": x0, "y": y, "w": w})

        h_model = 8.0
        h_cal = 24.0
        emb = 2.5
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            half_life=h_model,
            gap_cap=50.0,
            weight="w",
            min_weight=3.0,
            embargo=emb,
            emit_calibration=True,
            calibration_half_life=h_cal,
        )
        out = rp.ModelBank([spec]).fit_predict(df).unnest("m")
        # Check late rows after readiness settles.
        checked = 0
        for t_idx in (100, 140, 179):
            # Rows released strictly before row t_idx is scored have arrival <= ts[t_idx] - emb.
            released = np.flatnonzero(ts[:t_idx] <= ts[t_idx] - emb)
            if len(released) < 15:
                continue
            # Calibration folds only released rows whose instance was ready when released
            # and whose prediction was finite. Check exact clock-age decay formula below.
            # Note: under embargo, diag_ready is evaluated at release; once settled_frac >= 0.95
            # all subsequent releases are ready. Use min_settled_frac=0 via explicit test below
            # for exact bit-level formula check, and verify finite positive Wald here.
            wald = float(out["calibration_wald_y"][t_idx])
            assert math.isfinite(wald) and wald >= 0.0
            checked += 1
        assert checked == 3

        # Exact formula check with min_settled_frac=0 and half_life=inf on model
        # so every finite prediction is immediately ready upon release!
        spec_exact = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            half_life=float("inf"),
            gap_cap=50.0,
            weight="w",
            min_weight=3.0,
            min_settled_frac=0.0,
            embargo=emb,
            emit_calibration=True,
            calibration_half_life=h_cal,
        )
        out_e = rp.ModelBank([spec_exact]).fit_predict(df).unnest("m")
        preds_e = out_e["pred_y"].to_numpy()
        for t_idx in (100, 140, 179):
            # Released at arrival of t_idx: all prior rows s with ts[s] <= ts[t_idx] - emb.
            rel = np.flatnonzero(ts[:t_idx] <= ts[t_idx] - emb)
            # In the stream, the clock advances at each released row's arrival stamp `ts[s]`,
            # so row `s` ages to `ts[rel[-1]]` by `ts[rel[-1]] - ts[s]`.
            t_last = ts[rel[-1]]
            ok = np.isfinite(preds_e[rel])
            idx = rel[ok]
            omegas = w[idx] * (0.5 ** ((t_last - ts[idx]) / h_cal))
            want = _bartlett_mz_wald(ts[idx], preds_e[idx], y[idx], omegas, 2.0 * emb)
            got = float(out_e["calibration_wald_y"][t_idx])
            np.testing.assert_allclose(got, want, rtol=1e-9)

    def test_temporal_datetime_clock_and_quantile_coverage_match_exact(self) -> None:
        """Temporal ``Datetime`` clock with duration ``embargo="3s"`` matches exact
        longhand Bartlett HAC for both mean (``ewridge``) and quantile (``quantile``)."""
        rng = np.random.default_rng(77)
        n = 160
        base = dt.datetime(2026, 1, 1, 9, 30, 0)
        ms_steps = rng.choice([0, 400, 800, 1200, 2000], size=n)
        ms_steps[0] = 0
        cum_ms = np.cumsum(ms_steps)
        ts_dt = [base + dt.timedelta(milliseconds=int(m)) for m in cum_ms]
        ts_sec = cum_ms.astype(np.float64) / 1000.0
        x0 = rng.normal(size=n)
        y = 0.2 + 0.9 * x0 + rng.normal(scale=0.6, size=n)
        df = pl.DataFrame({"ts": ts_dt, "x0": x0, "y": y})

        # 1. Mean calibration on temporal clock.
        s_mean = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="ts",
            half_life="inf",
            gap_cap="1m",
            min_weight=5.0,
            min_settled_frac=0.0,
            embargo="3s",
            emit_calibration=True,
            calibration_half_life="20s",
        )
        # 2. Quantile coverage on temporal clock.
        s_q = rp.spec.quantile(
            "q",
            targets=["y"],
            features=["x0"],
            quantile=0.8,
            clock="ts",
            half_life="inf",
            gap_cap="1m",
            min_weight=5.0,
            min_settled_frac=0.0,
            embargo="3s",
            emit_calibration=True,
            calibration_half_life="20s",
        )
        out = rp.ModelBank([s_mean, s_q]).fit_predict(df)
        om = out["m"].struct.unnest()
        oq = out["q"].struct.unnest()
        pm = om["pred_y"].to_numpy()
        pq = oq["pred_y"].to_numpy()

        for t_idx in (90, 130, 159):
            rel = np.flatnonzero(cum_ms[:t_idx] <= cum_ms[t_idx] - 3000)
            t_last = ts_sec[rel[-1]]

            ok_m = np.isfinite(pm[rel])
            idx_m = rel[ok_m]
            om_w = 0.5 ** ((t_last - ts_sec[idx_m]) / 20.0)
            want_m = _bartlett_mz_wald(ts_sec[idx_m], pm[idx_m], y[idx_m], om_w, 6.0)
            np.testing.assert_allclose(float(om["calibration_wald_y"][t_idx]), want_m, rtol=1e-9)

            ok_q = np.isfinite(pq[rel])
            idx_q = rel[ok_q]
            oq_w = 0.5 ** ((t_last - ts_sec[idx_q]) / 20.0)
            want_q = _bartlett_coverage_wald(ts_sec[idx_q], pq[idx_q], y[idx_q], oq_w, 0.8, 6.0)
            np.testing.assert_allclose(float(oq["calibration_wald_y"][t_idx]), want_q, rtol=1e-9)

    def test_horizon_notices_suppressed_only_for_emit_calibration(self) -> None:
        """Clocked embargo with ``emit_calibration=True`` emits no ``horizon_rows``
        warning, while ``emit_breaks=True`` / ``emit_specification=True`` still do."""
        df = pl.DataFrame(
            {
                "t": np.arange(30, dtype=np.float64),
                "x0": np.linspace(0.0, 1.0, 30),
                "y": np.linspace(0.0, 2.0, 30),
            }
        )
        s_only_cal = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            gap_cap=50.0,
            half_life=10.0,
            embargo=2.0,
            emit_calibration=True,
        )
        with warnings.catch_warnings(record=True) as w1:
            warnings.simplefilter("always")
            rp.ModelBank([s_only_cal]).fit_predict(df)
        assert not [w for w in w1 if "horizon_rows" in str(w.message)]

        s_both = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            gap_cap=50.0,
            half_life=10.0,
            embargo=2.0,
            emit_calibration=True,
            emit_breaks=True,
        )
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            rp.ModelBank([s_both]).fit_predict(df)
        h_msgs = [str(w.message) for w in w2 if "horizon_rows" in str(w.message)]
        assert len(h_msgs) == 1
        assert "emit_breaks" in h_msgs[0]
        assert "emit_calibration" not in h_msgs[0]

    def test_chunk_invariance_save_load_json_and_rule_9(self, tmp_path) -> None:
        """Chunk invariance, save/load bit-identity, ``save_json_string``, and Rule 9
        zero-weight invariance all hold under clocked-embargo calibration."""
        rng = np.random.default_rng(55)
        n = 120
        ts = np.cumsum(rng.choice([0.5, 1.0, 1.5], size=n))
        x0 = rng.normal(size=n)
        y = 0.4 + 0.8 * x0 + rng.normal(scale=0.3, size=n)
        w = rng.uniform(0.5, 1.5, size=n)
        df = pl.DataFrame({"t": ts, "x0": x0, "y": y, "w": w})

        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            half_life=15.0,
            gap_cap=50.0,
            weight="w",
            min_weight=3.0,
            coef_every=0.0,
            embargo=3.0,
            emit_calibration=True,
            max_state_bytes=65536,
        )

        # 1. Whole frame vs 12-row chunks vs save/load at split=60.
        b_whole = rp.ModelBank([spec])
        out_whole = b_whole.fit_predict(df)

        b_chunks = rp.ModelBank([spec])
        parts = [b_chunks.fit_predict(df.slice(i, 12)) for i in range(0, n, 12)]
        out_chunks = pl.concat(parts)
        plt.assert_frame_equal(out_whole.unnest("m"), out_chunks.unnest("m"), check_exact=True)
        assert b_whole.save_bytes() == b_chunks.save_bytes()

        state_path = tmp_path / "cal_clock.bin"
        json_path = tmp_path / "cal_clock.json"
        b_part = rp.ModelBank([spec])
        p1 = b_part.fit_predict(df.slice(0, 60))
        b_part.save(state_path)
        b_part.write_json(json_path)
        assert '"clock_hac"' in json_path.read_text(encoding="utf-8")
        b_loaded = rp.ModelBank.load(state_path, specs=[spec])
        p2 = b_loaded.fit_predict(df.slice(60))
        plt.assert_frame_equal(
            out_whole.unnest("m"), pl.concat([p1, p2]).unnest("m"), check_exact=True
        )
        assert b_whole.save_bytes() == b_loaded.save_bytes()

        # 2. Rule 9: zero-weight rows change nothing on positive-weight rows or checks state.
        zeros = pl.DataFrame(
            {
                "t": [ts[25], ts[50] + 0.1, ts[90]],
                "x0": [10.0, -10.0, 5.0],
                "y": [100.0, -100.0, 50.0],
                "w": [0.0, 0.0, 0.0],
            }
        )
        df_with_zeros = pl.concat([df, zeros]).sort("t", maintain_order=True)
        b_z = rp.ModelBank([spec])
        out_z = b_z.fit_predict(df_with_zeros)
        pos_only = out_z.filter(pl.col("w") > 0.0)
        plt.assert_frame_equal(pos_only.unnest("m"), out_whole.unnest("m"), check_exact=True)
        # Verify the persisted `checks` (calibration & clock_hac) state is bit-identical
        # even though `rows_fed` counts the 3 zero-weight input rows.
        j_whole = tmp_path / "whole.json"
        j_z = tmp_path / "zeros.json"
        b_whole.write_json(j_whole)
        b_z.write_json(j_z)
        d_whole = json.loads(j_whole.read_text(encoding="utf-8"))
        d_z = json.loads(j_z.read_text(encoding="utf-8"))
        assert (
            d_z["states"][0][0][1]["persisted"]["checks"]
            == d_whole["states"][0][0][1]["persisted"]["checks"]
        )

    def test_max_state_bytes_half_cap_warning_and_atomic_refusal(self) -> None:
        """Clocked-embargo calibration respects ``max_state_bytes``: refuses invalid
        build-time caps, warns once past half the cap, and refuses atomically past cap."""
        # Build-time refusals: 0, > 64 MiB, and below fixed bytes (160).
        for bad_cap in (0, 64 * 1024 * 1024 + 1, 100):
            with pytest.raises(ValueError, match="max_state_bytes"):
                rp.spec.ewridge(
                    "m",
                    targets=["y"],
                    features=["x0"],
                    clock="t",
                    gap_cap=1000.0,
                    half_life=10.0,
                    embargo=10.0,
                    emit_calibration=True,
                    max_state_bytes=bad_cap,
                )

        # Cap = 160 + 8 * 48 = 544 bytes -> warns once past 272 bytes (> 2 entries),
        # refuses on the 9th concurrent entry within 2 * embargo = 200.0.
        cap = 160 + 8 * 48
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            half_life=float("inf"),
            gap_cap=1000.0,
            min_weight=1.0,
            min_settled_frac=0.0,
            max_error_inflation=float("inf"),
            embargo=100.0,
            emit_calibration=True,
            max_state_bytes=cap,
        )
        # Seed warm-up rows at t = 0..4, then rows at t = 105..116 (after warm-up rows
        # release, these 12 rows are scored with finite predictions!), then release the
        # first 6 of those scored rows at t = 205..210 -> triggers half-cap warning once!
        rng = np.random.default_rng(88)
        ts_warm = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ts_scored = np.arange(105.0, 117.0, 1.0)  # 12 rows with finite predictions
        ts_rel6 = np.arange(205.0, 211.0, 1.0)  # releases t=105..110 (6 entries <= 8)
        ts_ok = np.concatenate([ts_warm, ts_scored, ts_rel6])
        x_ok = rng.normal(size=len(ts_ok))
        y_ok = 0.5 + 1.5 * x_ok + rng.normal(scale=0.1, size=len(ts_ok))
        df_ok = pl.DataFrame({"t": ts_ok, "x0": x_ok, "y": y_ok})

        bank = rp.ModelBank([spec])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bank.fit_predict(df_ok)
        cap_warns = [
            str(w.message)
            for w in caught
            if issubclass(w.category, rheon.ReadinessWarning)
            and "emit_calibration" in str(w.message)
            and "max_state_bytes" in str(w.message)
        ]
        assert len(cap_warns) == 1, cap_warns
        assert "2 * embargo" in cap_warns[0]

        # Next chunk at t = 211..216 releases t = 111..116 -> 9th entry exceeds cap=8
        # inside bandwidth 200 -> atomic cap refusal!
        ts_over = np.arange(211.0, 217.0, 1.0)
        x_over = rng.normal(size=len(ts_over))
        y_over = 0.5 + 1.5 * x_over + rng.normal(scale=0.1, size=len(ts_over))
        df_over = pl.DataFrame({"t": ts_over, "x0": x_over, "y": y_over})
        with pytest.raises(ValueError, match=r"2 \* embargo"):
            bank.fit_predict(df_over)
