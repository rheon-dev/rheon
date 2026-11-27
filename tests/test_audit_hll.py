"""HyperLogLog++ and sliding-window HyperLogLog++ distinct counts in ``audit``.

Held against ``datasketch.HyperLogLogPlusPlus`` (with ``audit``'s SplitMix64
hash handed in via ``hashfunc``) on plain streams, row windows
(``distinct_window_rows``), and clock windows (``distinct_window_size``,
``closed="right"`` and ``"both"`` across float, integer, temporal, and row
clocks), plus Polars' ``n_unique`` on usable values.
"""

from __future__ import annotations

import struct
import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from datasketch import HyperLogLogPlusPlus

import ledger
from rheon import polars as rp

TIER = "mixed"

M64 = (1 << 64) - 1
INPUT_BOUND = 1e100


def _splitmix64_bytes(b: bytes) -> int:
    k = struct.unpack("<Q", b)[0]
    z = (k + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def _key_bytes(v: float) -> bytes:
    return struct.pack("<d", 0.0 if v == 0.0 else float(v))


def _usable(v: float | None) -> bool:
    return v is not None and np.isfinite(v) and abs(v) <= INPUT_BOUND


def _ds_count(values: list[float | None], p: int) -> float:
    h = HyperLogLogPlusPlus(p=p, hashfunc=_splitmix64_bytes)
    for v in values:
        if _usable(v):
            assert v is not None
            h.update(_key_bytes(v))
    return float(h.count())


class TestPlainHllAgainstDatasketch:
    @pytest.mark.parametrize("p", [8, 10, 12, 14, 16])
    @pytest.mark.parametrize("kind", ["random", "sequential", "grid"])
    def test_matches_datasketch_on_every_precision_and_distribution(self, p: int, kind: str):
        rng = np.random.default_rng(100 + p)
        n = 600 if p >= 14 else 1_200
        if kind == "random":
            vals = rng.normal(size=n)
        elif kind == "sequential":
            vals = np.arange(n, dtype=float)
        else:
            vals = np.round(rng.uniform(-50.0, 50.0, size=n), 2)
        df = pl.DataFrame({"x": vals})
        spec = rp.spec.audit("au", columns=["x"], distinct_precision=p)
        bank = rp.ModelBank([spec])
        raw = bank.fit_predict(df)
        assert raw["au"].struct.fields == ["approx_n_unique_x", "weight_sum"]
        out = raw.unnest("au")

        # Final estimate matches datasketch within 1e-12 relative.
        want = _ds_count(vals.tolist(), p)
        got = bank.audit().row(0, named=True)["approx_n_unique"]
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12)

        # Spot-check per-row out-of-sample values against incremental datasketch.
        ds = HyperLogLogPlusPlus(p=p, hashfunc=_splitmix64_bytes)
        emitted = out["approx_n_unique_x"].to_list()
        for i, v in enumerate(vals.tolist()):
            assert emitted[i] == pytest.approx(float(ds.count()), rel=1e-12, abs=1e-12), i
            ds.update(_key_bytes(v))

    def test_past_distinct_cap_reports_approx_beside_null_exact_within_3se(self):
        rng = np.random.default_rng(7)
        n = 12_000
        x = rng.normal(size=n)
        df = pl.DataFrame({"x": x})
        spec = rp.spec.audit("au", columns=["x"], distinct_cap=256, distinct_precision=12)
        bank = rp.ModelBank([spec])
        bank.fit(df)
        row = bank.audit().row(0, named=True)
        assert row["n_unique"] is None
        truth = df["x"].n_unique()
        se = 1.04 / np.sqrt(1 << 12)
        assert abs(row["approx_n_unique"] - truth) / truth < 3.0 * se

    def test_default_off_moves_no_schema_or_output(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 2.0]})
        bank = rp.ModelBank([rp.spec.audit("au", columns=["x"])])
        out = bank.fit_predict(df)
        assert out["au"].struct.fields == ["weight_sum"]
        assert "approx_n_unique" not in bank.audit().columns

    def test_unusable_values_and_signed_zero_match_n_unique(self):
        vals: list[float | None] = [
            0.0,
            -0.0,
            None,
            float("nan"),
            float("inf"),
            -float("inf"),
            1e200,
            -1e200,
            1.5,
            1.5,
            2.5,
        ]
        df = pl.DataFrame({"x": vals})
        spec = rp.spec.audit("au", columns=["x"], distinct_precision=10)
        bank = rp.ModelBank([spec])
        out = bank.fit_predict(df).unnest("au")
        row = bank.audit().row(0, named=True)
        assert row["n_unique"] == 3
        assert row["approx_n_unique"] == pytest.approx(_ds_count(vals, 10), rel=1e-12, abs=1e-12)
        assert out["approx_n_unique_x"][0] == 0.0
        # Row 2 (after 0.0 and -0.0) has the exact same estimate as row 1 (after 0.0).
        assert out["approx_n_unique_x"][2] == out["approx_n_unique_x"][1]


class TestSlidingWindowHllAgainstDatasketch:
    def test_distinct_window_rows_matches_datasketch_every_row(self):
        rng = np.random.default_rng(11)
        vals: list[float | None] = rng.integers(0, 60, size=140).astype(float).tolist()
        for idx in (12, 37, 81, 105):
            vals[idx] = None
        vals[50] = float("nan")
        vals[70] = float("inf")
        w = 25
        p = 8
        df = pl.DataFrame({"x": vals})
        spec = rp.spec.audit(
            "au",
            columns=["x"],
            distinct_precision=p,
            distinct_window_rows=w,
        )
        bank = rp.ModelBank([spec])
        out = bank.fit_predict(df).unnest("au")
        emitted = out["approx_n_unique_x"].to_list()
        for i in range(len(vals)):
            want = _ds_count(vals[max(0, i - w) : i], p)
            assert emitted[i] == pytest.approx(want, rel=1e-12, abs=1e-12), f"row {i}"
        final_want = _ds_count(vals[-w:], p)
        assert bank.audit().row(0, named=True)["approx_n_unique"] == pytest.approx(
            final_want, rel=1e-12, abs=1e-12
        )

    @pytest.mark.parametrize("closed", ["right", "both"])
    def test_distinct_window_size_float_clock_matches_datasketch_every_row(self, closed: str):
        rng = np.random.default_rng(21)
        n = 110
        vals: list[float | None] = rng.integers(0, 50, size=n).astype(float).tolist()
        vals[15] = None
        # Irregular float timestamps with exact boundary multiples and duplicates.
        ts = []
        t = 0.0
        for i in range(n):
            if i == 0:
                t = 0.0
            elif i % 9 == 0:
                pass  # duplicate timestamp
            elif i % 17 == 0:
                t += 10.0  # exact window edge
            elif i % 23 == 0:
                t += 15.0  # clears window
            else:
                t += 2.0
            ts.append(t)
        w = 10.0
        p = 8
        df = pl.DataFrame({"t": ts, "x": vals})
        spec = rp.spec.audit(
            "au",
            columns=["x"],
            clock="t",
            gap_cap=50.0,
            distinct_precision=p,
            distinct_window_size=w,
            closed=closed,
        )
        bank = rp.ModelBank([spec])
        out = bank.fit_predict(df).unnest("au")
        emitted = out["approx_n_unique_x"].to_list()

        def in_window(now: float, stamp: float) -> bool:
            return (now - stamp) < w if closed == "right" else (now - stamp) <= w

        for i in range(n):
            win_vals = [vals[j] for j in range(i) if in_window(ts[i], ts[j])]
            want = _ds_count(win_vals, p)
            assert emitted[i] == pytest.approx(want, rel=1e-12, abs=1e-12), (closed, i)
        final_vals = [vals[j] for j in range(n) if in_window(ts[-1], ts[j])]
        assert bank.audit().row(0, named=True)["approx_n_unique"] == pytest.approx(
            _ds_count(final_vals, p), rel=1e-12, abs=1e-12
        )

        # Scoring future timestamps via predict() evaluates the window at each
        # row's own stamp without mutating the bank's state.
        before_audit = bank.audit()
        future_ts = [ts[-1], ts[-1] + 5.0, ts[-1] + 10.0, ts[-1] + 25.0]
        scored = bank.predict(pl.DataFrame({"t": future_ts, "x": [0.0] * 4})).unnest("au")
        for idx, ft in enumerate(future_ts):
            f_vals = [vals[j] for j in range(n) if in_window(ft, ts[j])]
            assert scored["approx_n_unique_x"][idx] == pytest.approx(
                _ds_count(f_vals, p), rel=1e-12, abs=1e-12
            )
        assert bank.audit().equals(before_audit)

    @pytest.mark.parametrize("closed", ["right", "both"])
    def test_distinct_window_size_temporal_duration_matches_datasketch(self, closed: str):
        rng = np.random.default_rng(31)
        n = 80
        t0 = datetime(2025, 1, 1, 9, 30, 0)
        ts = [t0 + timedelta(seconds=5 * i) for i in range(n)]
        vals = rng.integers(0, 40, size=n).astype(float).tolist()
        df = pl.DataFrame({"t": ts, "x": vals}).with_columns(pl.col("t").cast(pl.Datetime("ms")))
        spec = rp.spec.audit(
            "au",
            columns=["x"],
            clock="t",
            gap_cap="5m",
            distinct_precision=8,
            distinct_window_size="30s",
            closed=closed,
        )
        bank = rp.ModelBank([spec])
        out = bank.fit_predict(df).unnest("au")
        emitted = out["approx_n_unique_x"].to_list()
        span = timedelta(seconds=30)
        for i in range(n):
            win = [
                vals[j]
                for j in range(i)
                if ((ts[i] - ts[j]) < span if closed == "right" else (ts[i] - ts[j]) <= span)
            ]
            assert emitted[i] == pytest.approx(_ds_count(win, 8), rel=1e-12, abs=1e-12), (closed, i)

    def test_distinct_window_size_row_clock_without_clock_column(self):
        rng = np.random.default_rng(41)
        vals = rng.integers(0, 30, size=60).astype(float).tolist()
        df = pl.DataFrame({"x": vals})
        spec = rp.spec.audit(
            "au",
            columns=["x"],
            distinct_precision=8,
            distinct_window_size=10.0,
            closed="both",
        )
        bank = rp.ModelBank([spec])
        out = bank.fit_predict(df).unnest("au")
        emitted = out["approx_n_unique_x"].to_list()
        for i in range(len(vals)):
            # On row clock with closed="both" and span=10: j < i with i - j <= 10.
            win = [vals[j] for j in range(i) if (i - j) <= 10]
            assert emitted[i] == pytest.approx(_ds_count(win, 8), rel=1e-12, abs=1e-12), i


class TestInvariantsMergeSaveAndCaps:
    @pytest.mark.parametrize(
        "mode",
        [
            {"distinct_precision": 8},
            {"distinct_precision": 8, "distinct_window_rows": 20},
            {"distinct_precision": 8, "distinct_window_size": 15.0, "closed": "both"},
        ],
    )
    def test_chunk_invariance_and_save_load_continuation(self, tmp_path, mode: dict):
        rng = np.random.default_rng(51)
        n = 150
        df = pl.DataFrame(
            {
                "t": np.cumsum(rng.choice([0.0, 1.0, 2.0, 4.0], size=n)),
                "g": np.where(np.arange(n) % 2 == 0, "a", "b"),
                "x": rng.integers(0, 80, size=n).astype(float),
                "y": rng.normal(size=n),
            }
        )
        spec = rp.spec.audit(
            "au",
            columns=["x", "y"],
            clock="t",
            gap_cap=50.0,
            group="g",
            **mode,
        )
        # 1 chunk
        b1 = rp.ModelBank([spec])
        out1 = b1.fit_predict(df)

        # Many chunks (size 7)
        b_many = rp.ModelBank([spec])
        parts = [b_many.fit_predict(s) for s in df.iter_slices(7)]
        out_many = pl.concat(parts)
        assert out_many.equals(out1)
        assert b_many.audit().equals(b1.audit())

        # Save at split 60, load, continue
        b_half = rp.ModelBank([spec])
        p1 = b_half.fit_predict(df.head(60))
        path = tmp_path / "audit_hll.state"
        b_half.save(path)
        loaded = rp.ModelBank.load(path)
        p2 = loaded.fit_predict(df.tail(n - 60))
        assert pl.concat([p1, p2]).equals(out1)
        assert loaded.audit().equals(b1.audit())

    def test_pooled_merge_plain_and_clock_window_and_refuses_row_window(self):
        rng = np.random.default_rng(61)
        n = 200
        df = pl.DataFrame(
            {
                "t": np.cumsum(rng.uniform(0.5, 2.0, size=n)),
                "g": np.where(np.arange(n) % 2 == 0, "a", "b"),
                "x": rng.integers(0, 120, size=n).astype(float),
            }
        )
        # Plain HLL pooled == ungrouped single stream.
        g_plain = rp.ModelBank(
            [rp.spec.audit("au", columns=["x"], group="g", distinct_precision=10)]
        )
        u_plain = rp.ModelBank([rp.spec.audit("au", columns=["x"], distinct_precision=10)])
        g_plain.fit(df)
        u_plain.fit(df)
        assert (
            g_plain.audit(pooled=True).row(0, named=True)["approx_n_unique"]
            == u_plain.audit().row(0, named=True)["approx_n_unique"]
        )

        # Clock-windowed HLL pooled == ungrouped single stream.
        g_clk = rp.ModelBank(
            [
                rp.spec.audit(
                    "au",
                    columns=["x"],
                    clock="t",
                    gap_cap=50.0,
                    group="g",
                    distinct_precision=8,
                    distinct_window_size=25.0,
                )
            ]
        )
        u_clk = rp.ModelBank(
            [
                rp.spec.audit(
                    "au",
                    columns=["x"],
                    clock="t",
                    gap_cap=50.0,
                    distinct_precision=8,
                    distinct_window_size=25.0,
                )
            ]
        )
        g_clk.fit(df)
        u_clk.fit(df)
        assert (
            g_clk.audit(pooled=True).row(0, named=True)["approx_n_unique"]
            == u_clk.audit().row(0, named=True)["approx_n_unique"]
        )

        # Row-windowed HLL refuses pooled=True cleanly.
        g_rows = rp.ModelBank(
            [
                rp.spec.audit(
                    "au",
                    columns=["x"],
                    group="g",
                    distinct_precision=8,
                    distinct_window_rows=30,
                )
            ]
        )
        g_rows.fit(df)
        with pytest.raises(ValueError, match="distinct_window_rows"):
            g_rows.audit(pooled=True)

    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            ({"distinct_precision": 7}, "distinct_precision must be from 8 to 16"),
            ({"distinct_precision": 17}, "distinct_precision must be from 8 to 16"),
            ({"distinct_precision": 300}, "distinct_precision must be from 8 to 16"),
            ({"distinct_precision": -1}, "distinct_precision"),
            ({"distinct_window_rows": 10}, "distinct_window_rows applies only when"),
            ({"distinct_window_size": 10.0}, "distinct_window_size applies only when"),
            (
                {"distinct_precision": 8, "distinct_window_rows": 0},
                "distinct_window_rows must be at least 1",
            ),
            (
                {"distinct_precision": 8, "distinct_window_size": 0.0},
                "distinct_window_size must be finite and above 0",
            ),
            (
                {"distinct_precision": 8, "distinct_window_rows": 10, "distinct_window_size": 10.0},
                "two windows",
            ),
            (
                {"distinct_precision": 8, "distinct_window_rows": 10, "closed": "both"},
                "distinct_window_size is not set",
            ),
            (
                {"distinct_precision": 8, "closed": "both"},
                "distinct_window_size is not set",
            ),
            (
                {"distinct_precision": 8, "distinct_window_size": "5m"},
                "no clock to measure it",
            ),
            (
                {"distinct_precision": 8, "max_state_bytes": 0},
                "max_state_bytes must be above 0",
            ),
            (
                {"distinct_precision": 8, "max_state_bytes": (64 << 20) + 1},
                "max_state_bytes may be at most",
            ),
            (
                {"distinct_precision": 12, "max_state_bytes": 10_000},
                "max_state_bytes of 10000",
            ),
            (
                {"max_state_bytes": 1_000},
                "max_state_bytes of 1000",
            ),
        ],
    )
    def test_spec_refusals(self, kw: dict, match: str):
        with pytest.raises(ValueError, match=match):
            rp.spec.audit("au", columns=["x"], **kw)

    def test_runtime_max_state_bytes_warns_and_refuses_cleanly(self):
        # 1 col, distinct_cap=16 (base 5,760) + p=8 sliding clock (fixed 2,048) = 7,808 B.
        # Cap 8,208 B allows ~10 live Pareto entries.
        spec = rp.spec.audit(
            "au",
            columns=["x"],
            clock="t",
            gap_cap=1_000.0,
            distinct_cap=16,
            distinct_precision=8,
            distinct_window_size=500.0,
            max_state_bytes=8_208,
        )
        df = pl.DataFrame(
            {"t": np.arange(500, dtype=float), "x": np.arange(1000, 1500, dtype=float)}
        )
        bank = rp.ModelBank([spec])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bank.fit_predict(df.head(5))
        assert any("max_state_bytes" in str(w.message) for w in caught)
        with pytest.raises(ValueError, match="max_state_bytes of 8208"):
            bank.fit_predict(df.tail(495))


# ---------------------------------------------------------------------------
# 50-seed oracle grid + ledger guard
# ---------------------------------------------------------------------------

GRID_SEEDS = range(50)


def _grid_series(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7_000 + seed)
    n = 1_500
    x = rng.normal(size=n)
    t = np.cumsum(rng.uniform(0.5, 2.0, size=n))
    return t, x


def _grid_rheon_outputs() -> list[tuple[float, float, float, float]]:
    out = []
    for seed in GRID_SEEDS:
        t, x = _grid_series(seed)
        df = pl.DataFrame({"t": t, "x": x})
        specs = [
            rp.spec.audit("p8", columns=["x"], distinct_precision=8),
            rp.spec.audit("p12", columns=["x"], distinct_precision=12),
            rp.spec.audit("wrows", columns=["x"], distinct_precision=12, distinct_window_rows=200),
            rp.spec.audit(
                "wclk",
                columns=["x"],
                clock="t",
                gap_cap=100.0,
                distinct_precision=12,
                distinct_window_size=50.0,
            ),
        ]
        bank = rp.ModelBank(specs)
        bank.fit(df)
        out.append(
            (
                float(bank.audit("p8").row(0, named=True)["approx_n_unique"]),
                float(bank.audit("p12").row(0, named=True)["approx_n_unique"]),
                float(bank.audit("wrows").row(0, named=True)["approx_n_unique"]),
                float(bank.audit("wclk").row(0, named=True)["approx_n_unique"]),
            )
        )
    return out


ledger.guard(
    "audit_hll_oracle",
    _grid_rheon_outputs,
    oracles=["datasketch"],
    bits=52,
)


@pytest.mark.oracle_grid(reason="50 seeds of 1,500 rows against datasketch plain and sliding HLL++")
def test_audit_hll_matches_datasketch_over_50_seeds():
    rheon_vals = _grid_rheon_outputs()
    p8_gaps, p12_gaps, wrows_gaps, wclk_gaps, truth_errs = [], [], [], [], []
    for seed, (got_p8, got_p12, got_wr, got_wc) in zip(GRID_SEEDS, rheon_vals, strict=True):
        t, x = _grid_series(seed)
        xs = x.tolist()
        want_p8 = _ds_count(xs, 8)
        want_p12 = _ds_count(xs, 12)
        want_wr = _ds_count(xs[-200:], 12)
        last_t = float(t[-1])
        win_x = [xs[j] for j in range(len(xs)) if (last_t - float(t[j])) < 50.0]
        want_wc = _ds_count(win_x, 12)
        p8_gaps.append(abs(got_p8 - want_p8) / want_p8)
        p12_gaps.append(abs(got_p12 - want_p12) / want_p12)
        wrows_gaps.append(abs(got_wr - want_wr) / want_wr)
        wclk_gaps.append(abs(got_wc - want_wc) / want_wc)
        truth = pl.Series("x", x).n_unique()
        truth_errs.append(abs(got_p12 - truth) / truth)

    ledger.record("audit_hll_oracle", "plain_p8_rel_gap", p8_gaps, floor=1e-12)
    ledger.record("audit_hll_oracle", "plain_p12_rel_gap", p12_gaps, floor=1e-12)
    ledger.record("audit_hll_oracle", "sliding_rows_p12_rel_gap", wrows_gaps, floor=1e-12)
    ledger.record("audit_hll_oracle", "sliding_clock_p12_rel_gap", wclk_gaps, floor=1e-12)
    ledger.record("audit_hll_oracle", "plain_p12_truth_rel_err", truth_errs)

    assert max(p8_gaps) < 1e-12
    assert max(p12_gaps) < 1e-12
    assert max(wrows_gaps) < 1e-12
    assert max(wclk_gaps) < 1e-12
    # 50 seeds at p=12: RSE = 1.04/64 ≈ 1.625%, worst of 50 well under 4 SE.
    assert max(truth_errs) < 4.0 * 1.04 / np.sqrt(1 << 12)
