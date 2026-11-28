"""Per-group step-back policy (``on_step_back="error" | "skip"``) across the
model bank, frame namespaces, CLI, ``with_windows``, ``merge``,
``refresh_time``, and ``bars``.

Every operator is held to its exact oracle:

- On the model bank, skipping stale rows gives the exact same predictions and
  model state on every non-stale row as filtering out ``t < cum_max(t)`` per
  group beforehand, and writes an all-null output row for every stale row.
- On ``with_windows``, skipping stale rows keeps 1-to-1 row alignment with
  null formula outputs on every stale row and bit-identical values on every
  non-stale row.
- On ``merge``, ``refresh_time``, and ``bars``, skipping stale rows drops
  those rows from the lagging group/input and equals the pre-filtered
  monotonic oracle to the bit.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import warnings
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from conftest import spec_toml
from rheon import polars as rp

TIER = "essential"


# ----------------------------------------------------------------- helpers


def _drop_stale(
    df: pl.DataFrame, clock: str, group: str | None = None
) -> tuple[pl.DataFrame, list[bool]]:
    """Return ``(monotonic_frame, is_stale_mask)`` where a row is stale iff its
    clock is strictly below the maximum clock accepted so far in its group."""
    last: dict[Any, Any] = {}
    stale: list[bool] = []
    clocks = df[clock].to_list()
    groups = df[group].to_list() if group is not None else [None] * df.height
    for g, t in zip(groups, clocks, strict=True):
        if t is None:
            stale.append(False)
            continue
        prev = last.get(g)
        if prev is not None and t < prev:
            stale.append(True)
        else:
            stale.append(False)
            last[g] = t
    keep = pl.Series("_keep", [not s for s in stale])
    return df.filter(keep), stale


def _step_back_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, rp.ReadinessWarning) and "on_step_back" in str(w.message)
    ]


def _bank_frame() -> pl.DataFrame:
    """Two interleaved groups on a float clock with stale rows in both groups."""
    rows: list[tuple[str, float, float, float, float]] = [
        ("a", 1.0, 1.0, -0.5, 2.0),
        ("b", 1.0, -1.0, 0.4, -0.8),
        ("a", 2.0, 0.5, 1.2, 1.8),
        ("b", 3.0, 0.2, -0.7, 0.1),
        ("a", 4.0, -0.3, 0.8, 0.9),
        ("a", 3.5, 9.0, 9.0, 99.0),  # stale in a (-0.5)
        ("b", 5.0, 1.1, 0.3, 2.3),
        ("a", 3.0, -8.0, 7.0, -77.0),  # stale in a (-1.0)
        ("a", 6.0, 0.8, -0.2, 1.4),
        ("b", 4.0, 5.0, -5.0, 88.0),  # stale in b (-1.0)
        ("b", 7.0, -0.6, 0.9, -0.2),
        ("a", 8.0, 1.3, 0.1, 2.7),
        ("b", 9.0, 0.4, -0.4, 0.5),
        ("a", 7.9, 4.0, 4.0, 55.0),  # stale in a (-0.1)
        ("a", 10.0, -0.9, 1.1, -0.4),
        ("b", 11.0, 0.7, 0.6, 1.9),
    ]
    return pl.DataFrame(
        {
            "g": [r[0] for r in rows],
            "t": [r[1] for r in rows],
            "x0": [r[2] for r in rows],
            "x1": [r[3] for r in rows],
            "y": [r[4] for r in rows],
        }
    )


def _assert_all_null_struct(series: pl.Series, stale_indices: list[int]) -> None:
    fields = series.struct.fields
    for idx in stale_indices:
        row = series[idx]
        assert isinstance(row, dict)
        for field in fields:
            assert row[field] is None, f"stale row {idx} field {field!r} was {row[field]!r}"


# ------------------------------------------------------------ default parity


def test_default_error_policy_is_bit_identical_across_all_surfaces(tmp_path: Path) -> None:
    """Explicit ``on_step_back='error'`` matches omitting it to the bit in
    outputs, warnings, and saved-state bytes across every operator."""
    df = pl.DataFrame(
        {
            "g": ["a", "b", "a", "b", "a", "b"],
            "t": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 3.0, 5.0, 7.0, 9.0, 11.0],
        }
    )

    # Bank
    s_default = rp.spec.ewridge(
        "m", targets=["y"], features=["x"], clock="t", half_life=5.0, gap_cap=20.0, group="g"
    )
    s_explicit = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x"],
        clock="t",
        half_life=5.0,
        gap_cap=20.0,
        group="g",
        on_step_back="error",
    )
    b1, b2 = rp.ModelBank([s_default]), rp.ModelBank([s_explicit])
    assert_frame_equal(b1.fit_predict(df), b2.fit_predict(df), check_exact=True)
    assert b1.save_bytes() == b2.save_bytes()

    # with_windows
    st1, st2 = tmp_path / "w1.state", tmp_path / "w2.state"
    expr = rp.ops.ewm_mean("x", half_life=5.0, window_size=10.0)
    w1 = rp.stream.with_windows(df, m=expr, clock="t", gap_cap=20.0, group="g", save_state=st1)
    w2 = rp.stream.with_windows(
        df, m=expr, clock="t", gap_cap=20.0, group="g", on_step_back="error", save_state=st2
    )
    assert_frame_equal(w1, w2, check_exact=True)
    assert st1.read_bytes() == st2.read_bytes()

    # merge
    a = df.filter(pl.col("g") == "a")
    b = df.filter(pl.col("g") == "b")
    ms1, ms2 = tmp_path / "m1.state", tmp_path / "m2.state"
    m1 = rp.stream.merge({"a": a, "b": b}, clock="t", group="g", save_state=ms1)
    m2 = rp.stream.merge(
        {"a": a, "b": b}, clock="t", group="g", on_step_back="error", save_state=ms2
    )
    assert_frame_equal(m1, m2, check_exact=True)
    assert ms1.read_bytes() == ms2.read_bytes()

    # refresh_time
    rt_df = pl.DataFrame(
        {
            "g": ["a", "a", "a", "a"],
            "s": ["x", "y", "x", "y"],
            "t": [1.0, 2.0, 3.0, 4.0],
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    rs1, rs2 = tmp_path / "r1.state", tmp_path / "r2.state"
    r1 = rp.stream.refresh_time(
        rt_df, series="s", names=["x", "y"], clock="t", value="v", group="g", save_state=rs1
    )
    r2 = rp.stream.refresh_time(
        rt_df,
        series="s",
        names=["x", "y"],
        clock="t",
        value="v",
        group="g",
        on_step_back="error",
        save_state=rs2,
    )
    assert_frame_equal(r1, r2, check_exact=True)
    assert rs1.read_bytes() == rs2.read_bytes()

    # bars
    bs1, bs2 = tmp_path / "b1.state", tmp_path / "b2.state"
    bar1 = rp.stream.bars(
        df,
        clock="t",
        kind="count",
        threshold=2,
        group="g",
        save_state=bs1,
        close=pl.col("x").last(),
    )
    bar2 = rp.stream.bars(
        df,
        clock="t",
        kind="count",
        threshold=2,
        group="g",
        on_step_back="error",
        save_state=bs2,
        close=pl.col("x").last(),
    )
    assert_frame_equal(bar1, bar2, check_exact=True)
    assert bs1.read_bytes() == bs2.read_bytes()


# ---------------------------------------------------------------------- bank


@pytest.mark.parametrize("kind", ["ewridge", "rls", "kalman", "ew_cov"])
def test_bank_skip_matches_prefiltered_oracle_to_the_bit(kind: str) -> None:
    """On non-stale rows the bank matches a pre-filtered monotonic oracle to
    the bit; stale rows are all-null, counted in ``clock_backwards``, and
    summarized in one ``ReadinessWarning`` per chunk."""
    df = _bank_frame()
    clean, stale = _drop_stale(df, "t", "g")
    stale_idx = [i for i, s in enumerate(stale) if s]
    keep_mask = pl.Series("_keep", [not s for s in stale])

    extra: dict[str, Any] = {}
    if kind == "kalman":
        extra = dict(targets=["y"], features=["x0", "x1"], coef_half_life=10.0)
    elif kind == "ew_cov":
        extra = dict(features=["x0", "x1"], stats=["mean", "var", "cov"])
    else:
        extra = dict(targets=["y"], features=["x0", "x1"])

    spec_skip = getattr(rp.spec, kind)(
        "m",
        clock="t",
        half_life=5.0,
        gap_cap=20.0,
        group="g",
        on_step_back="skip",
        min_weight=0.0,
        **extra,
    )
    spec_oracle = getattr(rp.spec, kind)(
        "m", clock="t", half_life=5.0, gap_cap=20.0, group="g", min_weight=0.0, **extra
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bank = rp.ModelBank([spec_skip])
        out = bank.fit_predict(df)

    oracle_bank = rp.ModelBank([spec_oracle])
    oracle_out = oracle_bank.fit_predict(clean)

    assert_frame_equal(out.filter(keep_mask), oracle_out, check_exact=True)
    _assert_all_null_struct(out["m"], stale_idx)

    # Summary counters
    summary = bank.summary().sort("group")
    assert summary["clock_backwards"].dtype == pl.Int64
    assert dict(
        zip(summary["group"].to_list(), summary["clock_backwards"].to_list(), strict=True)
    ) == {"a": 3, "b": 1}
    assert dict(
        zip(summary["group"].to_list(), summary["rows_skipped"].to_list(), strict=True)
    ) == {"a": 3, "b": 1}
    assert summary["resets"].to_list() == [0, 0]

    # One summary ReadinessWarning naming spec, total skipped (4), and affected groups (2)
    msgs = _step_back_warnings(caught)
    assert len(msgs) == 1, msgs
    assert 'spec "m"' in msgs[0]
    assert "4" in msgs[0]
    assert "2 groups" in msgs[0]

    # bank.check() surfaces step_back findings
    findings = bank.check().filter(pl.col("code") == "step_back").sort("group")
    assert findings["group"].to_list() == ["a", "b"]
    assert findings["value"].to_list() == [3.0, 1.0]

    # LazyFrame & DataFrame namespaces match ModelBank bit-for-bit
    with warnings.catch_warnings(record=True) as lazy_caught:
        warnings.simplefilter("always")
        lf_out = df.lazy().rheon.fit_predict([spec_skip]).collect()
        df_out = df.rheon.fit_predict([spec_skip])
    assert_frame_equal(lf_out, out, check_exact=True)
    assert_frame_equal(df_out, out, check_exact=True)
    assert len(_step_back_warnings(lazy_caught)) >= 2


def test_bank_zero_skips_emits_no_step_back_warning() -> None:
    df, _ = _drop_stale(_bank_frame(), "t", "g")
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        half_life=5.0,
        gap_cap=20.0,
        group="g",
        on_step_back="skip",
        min_weight=0.0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rp.ModelBank([spec]).fit_predict(df)
    assert _step_back_warnings(caught) == []


def test_bank_composition_with_restart_after_step_back_and_inclusive_edge() -> None:
    """Backward step < T skips; == T skips (inclusive!); > T restarts."""
    # Clock steps: 10 -> 15 -> 12 (-3 < 5: skip) -> 10 (-5 == 5: skip!)
    # -> 16 -> 10 (-6 > 5: restart!) -> 12.
    df = pl.DataFrame(
        {
            "t": [10.0, 15.0, 12.0, 10.0, 16.0, 10.0, 12.0],
            "x": [1.0, 2.0, 99.0, 88.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, -99.0, -88.0, 6.0, 8.0, 10.0],
        }
    )
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x"],
        clock="t",
        half_life=5.0,
        gap_cap=50.0,
        min_weight=0.0,
        restart_after_step_back=5.0,
        on_step_back="skip",
    )
    # Oracle: remove only rows 2 and 3 (steps -3 and -5 <= 5); keep row 5 (step -6 > 5, restarts)
    oracle_df = df.filter(pl.Series([True, True, False, False, True, True, True]))
    oracle_spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x"],
        clock="t",
        half_life=5.0,
        gap_cap=50.0,
        min_weight=0.0,
        restart_after_step_back=5.0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bank = rp.ModelBank([spec])
        got = bank.fit_predict(df)

    oracle_bank = rp.ModelBank([oracle_spec])
    want = oracle_bank.fit_predict(oracle_df)

    assert_frame_equal(
        got.filter(pl.Series([True, True, False, False, True, True, True])), want, check_exact=True
    )
    _assert_all_null_struct(got["m"], [2, 3])
    # Row 5 restarted, so weight_sum before learning row 5 is 0.0
    assert got["m"].struct.field("weight_sum")[5] == 0.0
    summary = bank.summary()
    assert summary["clock_backwards"][0] == 3  # 2 skipped + 1 reset
    assert summary["resets"][0] == 1
    assert summary["rows_skipped"][0] == 2
    msgs = _step_back_warnings(caught)
    assert len(msgs) == 1 and "2 stale rows" in msgs[0]


def test_bank_rule9_zero_weight_rows_and_embargo_label_clock_and_formula_targets() -> None:
    """Rule 9: zero-weight rows count in no clock; stale rows never enter
    embargo, label_clock, or formula-target resolution buffers."""
    # Rule 9 test:
    # t=10 (w=1) -> t=15 (w=0, does NOT advance clock!) -> t=12 (w=1, valid because 12 >= 10!)
    # -> t=11 (w=0, stale because 11 < 12 -> skipped!) -> t=9 (w=1, stale -> skipped!) -> t=14 (w=1)
    df = pl.DataFrame(
        {
            "t": [10.0, 15.0, 12.0, 11.0, 9.0, 14.0],
            "w": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            "x": [1.0, 2.0, 3.0, 99.0, 88.0, 4.0],
            "y": [2.0, 4.0, 6.0, -99.0, -88.0, 8.0],
        }
    )
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x"],
        clock="t",
        weight="w",
        half_life=5.0,
        gap_cap=50.0,
        min_weight=0.0,
        on_step_back="skip",
    )
    # Oracle without rows 3 and 4 (the two stale rows, while keeping row 1 at
    # t=15, w=0 and row 2 at t=12, w=1).
    keep = [True, True, True, False, False, True]
    oracle_spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x"],
        clock="t",
        weight="w",
        half_life=5.0,
        gap_cap=50.0,
        min_weight=0.0,
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        got = rp.ModelBank([spec]).fit_predict(df)
    want = rp.ModelBank([oracle_spec]).fit_predict(df.filter(pl.Series(keep)))
    assert_frame_equal(got.filter(pl.Series(keep)), want, check_exact=True)
    _assert_all_null_struct(got["m"], [3, 4])

    # Embargo + label_clock + formula target: stale row must not enter held queue
    emb_df = pl.DataFrame(
        {
            "t": [1.0, 2.0, 3.0, 2.5, 4.0, 5.0, 6.0, 7.0],
            "avail": [2.0, 3.0, 4.0, 2.6, 5.0, 6.0, 7.0, 8.0],
            "x": [1.0, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0, 7.0],
            "y": [2.0, 4.0, 6.0, -99.0, 8.0, 10.0, 12.0, 14.0],
        }
    )
    clean_emb, stale_emb = _drop_stale(emb_df, "t")
    keep_emb = pl.Series([not s for s in stale_emb])

    for extra_kw in [
        {"embargo": 1.5, "targets": ["y"]},
        {"label_clock": "avail", "targets": ["y"]},
        {
            "embargo": 2.0,
            "targets": [
                rp.ops.ewm_mean("y", half_life=2.0, window_size=2.0, reverse=True).alias("fwd")
            ],
        },
    ]:
        s_skip = rp.spec.ewridge(
            "m",
            features=["x"],
            clock="t",
            half_life=5.0,
            gap_cap=50.0,
            min_weight=0.0,
            on_step_back="skip",
            **extra_kw,
        )
        s_clean = rp.spec.ewridge(
            "m", features=["x"], clock="t", half_life=5.0, gap_cap=50.0, min_weight=0.0, **extra_kw
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            g = rp.ModelBank([s_skip]).fit_predict(emb_df)
        w = rp.ModelBank([s_clean]).fit_predict(clean_emb)
        assert_frame_equal(g.filter(keep_emb), w, check_exact=True)
        _assert_all_null_struct(g["m"], [3])


def test_bank_chunk_invariance_save_load_multi_group_isolation_and_predict() -> None:
    """Chunk invariance (1..N chunks), save/load across stale rows, multi-group
    isolation under parallel execution, and ``ModelBank.predict``."""
    # 12 groups: only group "g0" and "g3" have stale rows; other 10 groups are clean.
    rows = []
    for step in range(30):
        for g_idx in range(12):
            t = float(step * 2)
            if g_idx in (0, 3) and step in (7, 15, 22):
                t = float(step * 2 - 5)  # stale
            x0 = (step + g_idx) * 0.1
            y = 1.5 * x0 + 0.2 * g_idx
            rows.append((f"g{g_idx}", t, x0, y))
    df = pl.DataFrame(
        {
            "g": [r[0] for r in rows],
            "t": [r[1] for r in rows],
            "x0": [r[2] for r in rows],
            "y": [r[3] for r in rows],
        }
    )
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0"],
        clock="t",
        half_life=10.0,
        gap_cap=50.0,
        group="g",
        min_weight=0.0,
        coef_every=0.0,
        on_step_back="skip",
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        whole_bank = rp.ModelBank([spec])
        whole = whole_bank.fit_predict(df)
        whole_bytes = whole_bank.save_bytes()

        # Chunk invariance across chunk sizes 1, 7, 37 (splitting right on stale rows)
        for chunk_size in (1, 7, 37):
            b = rp.ModelBank([spec])
            parts = [
                b.fit_predict(df.slice(i, chunk_size)) for i in range(0, df.height, chunk_size)
            ]
            assert_frame_equal(pl.concat(parts), whole, check_exact=True)
            assert b.save_bytes() == whole_bytes

        # Save/load mid-stream right before a stale row
        cut = 7 * 12
        b1 = rp.ModelBank([spec])
        p1 = b1.fit_predict(df.slice(0, cut))
        b2 = rp.ModelBank.load_bytes(b1.save_bytes())
        p2 = b2.fit_predict(df.slice(cut))
        assert_frame_equal(pl.concat([p1, p2]), whole, check_exact=True)
        assert b2.save_bytes() == whole_bytes

    # Multi-group isolation: clean groups match a bank fed only clean groups to the bit
    clean_groups = df.filter(~pl.col("g").is_in(["g0", "g3"]))
    clean_spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0"],
        clock="t",
        half_life=10.0,
        gap_cap=50.0,
        group="g",
        min_weight=0.0,
        coef_every=0.0,
    )
    clean_out = rp.ModelBank([clean_spec]).fit_predict(clean_groups)
    assert_frame_equal(whole.filter(~pl.col("g").is_in(["g0", "g3"])), clean_out, check_exact=True)

    # predict() scores rows at earlier clocks cleanly without skipping
    early = pl.DataFrame({"g": ["g0", "g1"], "t": [1.0, 1.0], "x0": [0.5, 0.5], "y": [0.0, 0.0]})
    pred = whole_bank.predict(early)
    assert pred["m"].struct.field("pred_y").null_count() == 0


def test_bank_numeric_int_and_temporal_clocks() -> None:
    """Int64, Datetime, Date, and Duration clocks all work identically."""
    base = dt.datetime(2025, 1, 1, 9, 30, 0)
    d0 = dt.date(2025, 1, 1)
    cases = [
        (
            "Int64",
            pl.Series("t", [10, 20, 15, 30, 25, 40], dtype=pl.Int64),
            dict(half_life=10.0, gap_cap=100.0, restart_after_step_back=8.0),
        ),
        (
            "Datetime",
            pl.Series("t", [base + dt.timedelta(seconds=s) for s in (10, 20, 15, 30, 25, 40)]),
            dict(half_life="10s", gap_cap="100s", restart_after_step_back="8s"),
        ),
        (
            "Date",
            pl.Series("t", [d0 + dt.timedelta(days=d) for d in (1, 3, 2, 5, 4, 7)]),
            dict(half_life="5d", gap_cap="30d", restart_after_step_back="2d"),
        ),
        (
            "Duration",
            pl.Series("t", [dt.timedelta(seconds=s) for s in (10, 20, 15, 30, 25, 40)]),
            dict(half_life="10s", gap_cap="100s", restart_after_step_back="8s"),
        ),
    ]
    for label, col, kw in cases:
        df = pl.DataFrame(
            {
                "t": col,
                "x": [1.0, 2.0, 99.0, 3.0, 88.0, 4.0],
                "y": [2.0, 4.0, -99.0, 6.0, -88.0, 8.0],
            }
        )
        clean, stale = _drop_stale(df, "t")
        keep = pl.Series([not s for s in stale])
        spec_skip = rp.spec.ewridge(
            "m", targets=["y"], features=["x"], clock="t", min_weight=0.0, on_step_back="skip", **kw
        )
        spec_clean = rp.spec.ewridge(
            "m", targets=["y"], features=["x"], clock="t", min_weight=0.0, **kw
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = rp.ModelBank([spec_skip]).fit_predict(df)
        want = rp.ModelBank([spec_clean]).fit_predict(clean)
        assert_frame_equal(got.filter(keep), want, check_exact=True)
        _assert_all_null_struct(got["m"], [2, 4])
        assert len(_step_back_warnings(caught)) == 1, label


def test_bank_cli_toml_matches_python_bit_for_bit(rheon_cli: Path, tmp_path: Path) -> None:
    """CLI TOML config with ``on_step_back = 'skip'`` matches ModelBank to the
    bit (outputs and saved state) and reports the step-back warning."""
    df = _bank_frame()
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        half_life=5.0,
        gap_cap=20.0,
        group="g",
        min_weight=0.0,
        coef_every=0.0,
        on_step_back="skip",
    )
    inp = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    st = tmp_path / "cli.state"
    cfg = tmp_path / "bank.toml"
    df.write_parquet(inp)
    cfg.write_text(
        f'input = "{inp.as_posix()}"\n'
        f'output = "{out.as_posix()}"\n'
        f'save_state = "{st.as_posix()}"\n'
        "chunk_size = 5\n\n" + spec_toml(spec),
        encoding="utf-8",
    )
    res = subprocess.run(
        [str(rheon_cli), "--config", str(cfg)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "on_step_back" in res.stderr

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        bank = rp.ModelBank([spec])
        expected = bank.fit_predict(df)

    assert_frame_equal(pl.read_parquet(out), expected, check_exact=True)
    assert st.read_bytes() == bank.save_bytes()


# --------------------------------------------------------------- with_windows


def test_with_windows_skip_matches_oracle_1to1_and_non_bare_formulas(tmp_path: Path) -> None:
    """``with_windows(..., on_step_back='skip')`` keeps 1-to-1 row alignment
    with null formula outputs (even for ``.fill_null(0.0)`` and compound
    expressions!), matches the pre-filtered oracle on non-stale rows across
    backward, forward, cross, profile, and row operators, inherits via
    ``like=``, is chunk-invariant, and resumes across held stale rows."""
    df = pl.DataFrame(
        {
            "g": ["a", "b", "a", "a", "b", "b", "a", "b", "a", "a", "b"],
            "t": [1.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 6.0, 4.5, 8.0, 9.0],
            "x": [10.0, 100.0, 20.0, 999.0, 200.0, 888.0, 30.0, 300.0, 777.0, 40.0, 400.0],
        }
    )
    clean, stale = _drop_stale(df, "t", "g")
    stale_idx = [i for i, s in enumerate(stale) if s]
    keep = pl.Series([not s for s in stale])

    formulas = {
        "back": rp.ops.ewm_mean("x", half_life=4.0, window_size=10.0),
        "ahead": rp.ops.ewm_mean("x", half_life=4.0, window_size=5.0, reverse=True),
        "filled": rp.ops.ewm_mean("x", half_life=4.0).fill_null(0.0),
        "compound": pl.col("x") + rp.ops.ewm_mean("x", half_life=4.0),
        "step": rp.ops.increment("x"),
        "sign": rp.ops.tick_sign("x"),
        "cadence": rp.ops.clock_step("t", half_life=4.0, stat="mean"),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = rp.stream.with_windows(
            df, clock="t", gap_cap=50.0, group="g", on_step_back="skip", **formulas
        )
    want = rp.stream.with_windows(clean, clock="t", gap_cap=50.0, group="g", **formulas)

    assert got.height == df.height
    assert_frame_equal(got.select(["g", "t", "x"]), df, check_exact=True)
    assert_frame_equal(got.filter(keep), want, check_exact=True)
    for col in formulas:
        for idx in stale_idx:
            assert got[col][idx] is None, f"stale row {idx} col {col!r} = {got[col][idx]!r}"

    msgs = _step_back_warnings(caught)
    assert len(msgs) == 1
    assert "with_windows" in msgs[0] and "3 stale rows" in msgs[0] and "2 groups" in msgs[0]

    # Chunk invariance + LazyFrame/DataFrame namespace parity
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        for cs in (1, 2, 4):
            chunked = rp.stream.with_windows(
                df.lazy(),
                clock="t",
                gap_cap=50.0,
                group="g",
                on_step_back="skip",
                chunk_size=cs,
                **formulas,
            ).collect()
            assert_frame_equal(chunked, got, check_exact=True)
        ns_df = df.rheon.with_windows(
            clock="t", gap_cap=50.0, group="g", on_step_back="skip", **formulas
        )
        ns_lf = (
            df.lazy()
            .rheon.with_windows(clock="t", gap_cap=50.0, group="g", on_step_back="skip", **formulas)
            .collect()
        )
        assert_frame_equal(ns_df, got, check_exact=True)
        assert_frame_equal(ns_lf, got, check_exact=True)

    # Save/load continuation across held forward window containing a stale row
    st = tmp_path / "win.state"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        p1 = rp.stream.with_windows(
            df.slice(0, 6),
            clock="t",
            gap_cap=50.0,
            group="g",
            on_step_back="skip",
            save_state=st,
            **formulas,
        )
        p2 = rp.stream.with_windows(
            df.slice(6),
            clock="t",
            gap_cap=50.0,
            group="g",
            on_step_back="skip",
            load_state=st,
            **formulas,
        )
    assert_frame_equal(pl.concat([p1, p2]), got, check_exact=True)

    # like= inherits on_step_back="skip" from spec
    spec = rp.spec.ew_cov(
        "m",
        features=["x"],
        stats=["mean"],
        clock="t",
        half_life=4.0,
        gap_cap=50.0,
        group="g",
        on_step_back="skip",
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        via_like = rp.stream.with_windows(df, like=spec, **formulas)
    assert_frame_equal(via_like, got, check_exact=True)


def test_with_windows_cross_profile_and_restart_composition() -> None:
    """Cross-group and periodic profile operators, plus composition with
    ``restart_after_step_back``."""
    # Cross-group operator: stale row on group "a" must not enter stamp's cross set
    cdf = pl.DataFrame(
        {
            "g": ["a", "b", "a", "b", "a", "a", "b"],
            "t": [1.0, 1.0, 2.0, 2.0, 1.5, 3.0, 3.0],
            "x": [1.0, 2.0, 3.0, 4.0, 99.0, 5.0, 6.0],
        }
    )
    clean_c, stale_c = _drop_stale(cdf, "t", "g")
    keep_c = pl.Series([not s for s in stale_c])
    cross_expr = rp.ops.cross_rank("x")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        got_c = rp.stream.with_windows(
            cdf, r=cross_expr, clock="t", gap_cap=50.0, group="g", on_step_back="skip"
        )
    want_c = rp.stream.with_windows(clean_c, r=cross_expr, clock="t", gap_cap=50.0, group="g")
    assert_frame_equal(got_c.filter(keep_c), want_c, check_exact=True)
    assert got_c["r"][4] is None

    # Composition with restart_after_step_back=5.0: -3 skips, -5 skips, -6 restarts
    rdf = pl.DataFrame(
        {
            "t": [10.0, 15.0, 12.0, 10.0, 16.0, 10.0, 12.0],
            "x": [1.0, 2.0, 99.0, 88.0, 3.0, 4.0, 5.0],
        }
    )
    expr = rp.ops.ewm_sum("x", half_life=100.0)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        got_r = rp.stream.with_windows(
            rdf, s=expr, clock="t", gap_cap=100.0, restart_after_step_back=5.0, on_step_back="skip"
        )
    oracle_r = rp.stream.with_windows(
        rdf.filter(pl.Series([True, True, False, False, True, True, True])),
        s=expr,
        clock="t",
        gap_cap=100.0,
        restart_after_step_back=5.0,
    )
    assert_frame_equal(
        got_r.filter(pl.Series([True, True, False, False, True, True, True])),
        oracle_r,
        check_exact=True,
    )
    assert got_r["s"][2] is None and got_r["s"][3] is None


# --------------------------------------------------------------------- merge


def test_merge_skip_matches_oracle_chunks_save_load_restart_and_temporal(tmp_path: Path) -> None:
    """``rp.stream.merge(..., on_step_back='skip')`` drops stale rows, matches
    pre-filtered monotonic inputs to the bit, handles cross-input merged-group
    step-backs on resume, composes with ``restart_after_step_back``, is chunk
    invariant, resumes across state save/load, and works on temporal clocks."""
    # Each input is clock-ordered except for injected stale rows.
    a = pl.DataFrame(
        {
            "g": ["x", "x", "x", "y", "x", "y", "x"],
            "t": [1.0, 4.0, 2.0, 5.0, 6.0, 4.5, 8.0],
            "v": [10.0, 20.0, 99.0, 100.0, 30.0, 88.0, 40.0],
        }
    )
    b = pl.DataFrame(
        {
            "g": ["x", "y", "y", "x", "y"],
            "t": [3.0, 5.5, 2.5, 7.0, 9.0],
            "v": [1.0, 2.0, 77.0, 3.0, 4.0],
        }
    )
    clean_a, _ = _drop_stale(a, "t", None)
    clean_b, _ = _drop_stale(b, "t", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = rp.stream.merge(
            {"a": a, "b": b}, clock="t", group="g", carry=["v"], on_step_back="skip"
        )
    want = rp.stream.merge({"a": clean_a, "b": clean_b}, clock="t", group="g", carry=["v"])
    assert_frame_equal(got, want, check_exact=True)

    msgs = _step_back_warnings(caught)
    assert len(msgs) == 1 and "merge" in msgs[0] and "3 stale rows" in msgs[0]

    # Chunk invariance
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        for cs in (1, 2, 3):
            c_out = rp.stream.merge(
                {"a": a.lazy(), "b": b.lazy()},
                clock="t",
                group="g",
                carry=["v"],
                on_step_back="skip",
                chunk_size=cs,
            ).collect()
            assert_frame_equal(c_out, got, check_exact=True)

    # Save/load continuation
    st = tmp_path / "merge.state"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        p1 = rp.stream.merge(
            {"a": a.slice(0, 4), "b": b.slice(0, 2)},
            clock="t",
            group="g",
            carry=["v"],
            on_step_back="skip",
            save_state=st,
        )
        p2 = rp.stream.merge(
            {"a": a.slice(4), "b": b.slice(2)},
            clock="t",
            group="g",
            carry=["v"],
            on_step_back="skip",
            load_state=st,
        )
    assert_frame_equal(pl.concat([p1, p2]), got, check_exact=True)

    # Cross-input merged-group step-back on resume: in run 1 both inputs pass
    # t=10.0 so merged group "x" emits t=10.0. In run 2 input b steps back from
    # 11.0 to 8.0 (-3.0 > restart 2.0, so input b restarts), yet for merged
    # group "x" (last emitted 10.0) the step back is 2.0 <= 2.0 -> dropped as
    # stale by pull()!
    st_cross = tmp_path / "merge_cross.state"
    a1 = pl.DataFrame({"g": ["x", "y"], "t": [10.0, 12.0], "v": [1.0, 2.0]})
    b1 = pl.DataFrame({"g": ["y"], "t": [11.0], "v": [10.0]})
    b2 = pl.DataFrame({"g": ["x", "y", "x"], "t": [8.0, 13.0, 14.0], "v": [99.0, 20.0, 30.0]})
    a2 = pl.DataFrame({"g": ["y"], "t": [13.0], "v": [3.0]})
    with warnings.catch_warnings(record=True) as cross_caught:
        warnings.simplefilter("always")
        rp.stream.merge(
            {"a": a1, "b": b1},
            clock="t",
            group="g",
            restart_after_step_back=2.0,
            on_step_back="skip",
            save_state=st_cross,
        )
        resumed = rp.stream.merge(
            {"a": a2, "b": b2},
            clock="t",
            group="g",
            restart_after_step_back=2.0,
            on_step_back="skip",
            load_state=st_cross,
        )
    assert 99.0 not in resumed["v"].to_list()
    assert resumed.filter(pl.col("g") == "x")["v"].to_list() == [30.0]
    assert len(_step_back_warnings(cross_caught)) == 1

    # Composition with restart_after_step_back=5.0 + temporal Datetime clock
    t0 = dt.datetime(2025, 1, 1, 10, 0, 0)
    ta = pl.DataFrame(
        {
            "t": [t0 + dt.timedelta(seconds=s) for s in (10, 15, 12, 10, 16, 10, 12)],
            "v": [1.0, 2.0, 99.0, 88.0, 3.0, 4.0, 5.0],
        }
    )
    tb = pl.DataFrame({"t": [t0 + dt.timedelta(seconds=s) for s in (11, 17)], "v": [10.0, 20.0]})
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        got_t = rp.stream.merge(
            {"a": ta, "b": tb}, clock="t", restart_after_step_back="5s", on_step_back="skip"
        )
    ta_oracle = ta.filter(pl.Series([True, True, False, False, True, True, True]))
    want_t = rp.stream.merge({"a": ta_oracle, "b": tb}, clock="t", restart_after_step_back="5s")
    assert_frame_equal(got_t, want_t, check_exact=True)


# -------------------------------------------------------------- refresh_time


@pytest.mark.parametrize("pairs", [False, True])
def test_refresh_time_skip_matches_oracle_chunks_save_load_and_temporal(
    pairs: bool, tmp_path: Path
) -> None:
    """``rp.stream.refresh_time(..., on_step_back='skip')`` drops stale rows
    per group, matches the pre-filtered oracle to the bit for ``pairs``,
    ``age``, chunking, save/load, and temporal clocks."""
    t0 = dt.datetime(2025, 1, 1, 9, 0, 0)
    df = pl.DataFrame(
        {
            "g": ["g1", "g1", "g1", "g2", "g1", "g2", "g1", "g1", "g2", "g2", "g1", "g2"],
            "s": ["a", "b", "a", "a", "b", "b", "a", "b", "a", "b", "a", "a"],
            "t": [t0 + dt.timedelta(seconds=s) for s in (1, 2, 1, 1, 3, 2, 2, 4, 1, 5, 6, 7)],
            "v": [1.0, 2.0, 99.0, 10.0, 3.0, 20.0, 88.0, 4.0, 77.0, 30.0, 5.0, 40.0],
            "tag": list(range(12)),
        }
    )
    clean, _ = _drop_stale(df, "t", "g")

    kw: dict[str, Any] = dict(
        series="s",
        names=["a", "b"],
        clock="t",
        value="v",
        group="g",
        pairs=pairs,
        age=True,
        keep=["tag"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = rp.stream.refresh_time(df, on_step_back="skip", **kw)
    want = rp.stream.refresh_time(clean, **kw)
    assert_frame_equal(got, want, check_exact=True)

    msgs = _step_back_warnings(caught)
    assert len(msgs) == 1 and "refresh_time" in msgs[0] and "3 stale rows" in msgs[0]

    # Chunk invariance
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        for cs in (1, 2, 5):
            c_out = rp.stream.refresh_time(
                df.lazy(), on_step_back="skip", chunk_size=cs, **kw
            ).collect()
            assert_frame_equal(c_out, got, check_exact=True)

    # Save/load continuation
    st = tmp_path / f"rt_{pairs}.state"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        p1 = rp.stream.refresh_time(df.slice(0, 6), on_step_back="skip", save_state=st, **kw)
        p2 = rp.stream.refresh_time(df.slice(6), on_step_back="skip", load_state=st, **kw)
    assert_frame_equal(pl.concat([p1, p2]), got, check_exact=True)


# ---------------------------------------------------------------------- bars


@pytest.mark.parametrize("kind", ["count", "sum", "imbalance"])
def test_bars_skip_matches_oracle_chunks_save_load_and_restart(kind: str, tmp_path: Path) -> None:
    """``rp.stream.bars(..., on_step_back='skip')`` drops stale rows per group,
    matches the pre-filtered monotonic oracle across count/sum/imbalance,
    weights, ``final=True``, chunking, save/load, and ``restart_after_step_back``."""
    df = pl.DataFrame(
        {
            "g": ["a", "b", "a", "a", "a", "b", "b", "a", "a", "b", "b", "a"],
            "t": [1, 1, 3, 2, 5, 4, 6, 4, 8, 9, 7, 10],
            "p": [10.0, 20.0, 11.0, 99.0, 12.0, 21.0, 22.0, 88.0, 13.0, 23.0, 77.0, 14.0],
            "q": [2.0, 3.0, 2.0, 50.0, 2.0, 3.0, 3.0, 50.0, 2.0, 3.0, 50.0, 2.0],
            "s": [1.0, -1.0, 1.0, -50.0, 1.0, -1.0, -1.0, -50.0, 1.0, -1.0, -50.0, 1.0],
            "w": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    clean, _ = _drop_stale(df, "t", "g")
    kw: dict[str, Any] = dict(
        clock="t",
        kind=kind,
        threshold=2 if kind == "count" else (4.0 if kind == "sum" else 2.0),
        quantity=None if kind == "count" else ("q" if kind == "sum" else "s"),
        group="g",
        weight="w",
        final=True,
        open=pl.col("p").first(),
        close=pl.col("p").last(),
        rows=pl.len(),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = rp.stream.bars(df, on_step_back="skip", **kw)
    want = rp.stream.bars(clean, **kw)
    assert_frame_equal(got, want, check_exact=True)

    msgs = _step_back_warnings(caught)
    assert len(msgs) == 1 and "bars" in msgs[0] and "3 stale rows" in msgs[0]

    # Chunk invariance
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        for cs in (1, 3, 5):
            c_out = rp.stream.bars(df.lazy(), on_step_back="skip", chunk_size=cs, **kw).collect()
            assert_frame_equal(c_out, got, check_exact=True)

    # Save/load continuation
    st = tmp_path / f"bars_{kind}.state"
    kw_no_final = {**kw, "final": False}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        full_nf = rp.stream.bars(df, on_step_back="skip", **kw_no_final)
        p1 = rp.stream.bars(df.slice(0, 6), on_step_back="skip", save_state=st, **kw_no_final)
        p2 = rp.stream.bars(df.slice(6), on_step_back="skip", load_state=st, **kw_no_final)
    assert_frame_equal(pl.concat([p1, p2]), full_nf, check_exact=True)

    # Composition with restart_after_step_back=5 on bars
    rdf = pl.DataFrame(
        {
            "t": [10.0, 15.0, 12.0, 10.0, 16.0, 10.0, 12.0, 14.0],
            "p": [1.0, 2.0, 99.0, 88.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        got_r = rp.stream.bars(
            rdf,
            clock="t",
            kind="count",
            threshold=2,
            restart_after_step_back=5.0,
            on_step_back="skip",
            final=True,
            close=pl.col("p").last(),
        )
    oracle_rdf = rdf.filter(pl.Series([True, True, False, False, True, True, True, True]))
    want_r = rp.stream.bars(
        oracle_rdf,
        clock="t",
        kind="count",
        threshold=2,
        restart_after_step_back=5.0,
        final=True,
        close=pl.col("p").last(),
    )
    assert_frame_equal(got_r, want_r, check_exact=True)


# ---------------------------------------------------- validation & refusals


def test_validation_and_state_mismatch_refusals(tmp_path: Path) -> None:
    """Invalid words, wrong types, missing clock, ``like=`` clash, and state
    load policy mismatches are all refused cleanly across every surface."""
    df = pl.DataFrame({"t": [1.0, 2.0], "s": ["a", "b"], "x": [1.0, 2.0], "y": [1.0, 2.0]})

    # Spec validation
    with pytest.raises(ValueError, match="on_step_back"):
        rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x"],
            clock="t",
            half_life=5.0,
            gap_cap=10.0,
            on_step_back="drop",
        )  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="on_step_back"):
        rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x"],
            clock="t",
            half_life=5.0,
            gap_cap=10.0,
            on_step_back=123,
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="clock"):
        rp.spec.ewridge("m", targets=["y"], features=["x"], half_life=5.0, on_step_back="skip")
    # Default "error" without clock is allowed
    rp.spec.ewridge("m", targets=["y"], features=["x"], half_life=5.0, on_step_back="error")

    # with_windows validation
    expr = rp.ops.ewm_mean("x", half_life=5.0)
    with pytest.raises(ValueError, match="on_step_back"):
        rp.stream.with_windows(df, m=expr, clock="t", gap_cap=10.0, on_step_back="ignore")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="on_step_back"):
        rp.stream.with_windows(df, m=expr, clock="t", gap_cap=10.0, on_step_back=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="on_step_back"):
        rp.stream.with_windows(df, m=expr, clock="t", gap_cap=10.0, on_step_back=pl.col("x"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="clock"):
        rp.stream.with_windows(df, m=expr, on_step_back="skip")
    # Default "error" without clock is allowed
    rp.stream.with_windows(df, m=expr, on_step_back="error")
    # Clash with like=
    spec = rp.spec.ewridge(
        "m", targets=["y"], features=["x"], clock="t", half_life=5.0, gap_cap=10.0
    )
    with pytest.raises(rp.RefusedValueError, match="on_step_back"):
        rp.stream.with_windows(df, m=expr, like=spec, on_step_back="skip")

    # merge / refresh_time / bars invalid word & wrong type
    for bad, exc in [("drop", ValueError), (123, TypeError)]:
        with pytest.raises(exc, match="on_step_back"):
            rp.stream.merge({"a": df}, clock="t", on_step_back=bad)  # type: ignore[arg-type]
        with pytest.raises(exc, match="on_step_back"):
            rp.stream.refresh_time(
                df, series="s", names=["a", "b"], clock="t", value="x", on_step_back=bad
            )  # type: ignore[arg-type]
        with pytest.raises(exc, match="on_step_back"):
            rp.stream.bars(df, clock="t", kind="count", threshold=2, on_step_back=bad)  # type: ignore[arg-type]

    # State load mismatch on all 5 surfaces
    s_skip = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x"],
        clock="t",
        half_life=5.0,
        gap_cap=10.0,
        on_step_back="skip",
    )
    s_err = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x"],
        clock="t",
        half_life=5.0,
        gap_cap=10.0,
        on_step_back="error",
    )
    b = rp.ModelBank([s_skip])
    b.fit_predict(df)
    with pytest.raises(ValueError, match="on_step_back"):
        rp.ModelBank.load_bytes(b.save_bytes(), specs=[s_err])

    wst = tmp_path / "m_win.state"
    rp.stream.with_windows(df, m=expr, clock="t", gap_cap=10.0, on_step_back="skip", save_state=wst)
    with pytest.raises(ValueError, match="on_step_back"):
        rp.stream.with_windows(
            df, m=expr, clock="t", gap_cap=10.0, on_step_back="error", load_state=wst
        )

    mst = tmp_path / "m_merge.state"
    rp.stream.merge({"a": df}, clock="t", on_step_back="skip", save_state=mst)
    with pytest.raises(ValueError, match="on_step_back"):
        rp.stream.merge({"a": df}, clock="t", on_step_back="error", load_state=mst)

    rst = tmp_path / "m_rt.state"
    rp.stream.refresh_time(
        df, series="s", names=["a", "b"], clock="t", value="x", on_step_back="skip", save_state=rst
    )
    with pytest.raises(ValueError, match="on_step_back"):
        rp.stream.refresh_time(
            df,
            series="s",
            names=["a", "b"],
            clock="t",
            value="x",
            on_step_back="error",
            load_state=rst,
        )

    bst = tmp_path / "m_bars.state"
    rp.stream.bars(df, clock="t", kind="count", threshold=2, on_step_back="skip", save_state=bst)
    with pytest.raises(ValueError, match="on_step_back"):
        rp.stream.bars(
            df, clock="t", kind="count", threshold=2, on_step_back="error", load_state=bst
        )
