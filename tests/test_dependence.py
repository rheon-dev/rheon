"""`dependence` against scikit-learn and scipy oracles.

Covers:
- Plug-in binned mutual information against ``sklearn.metrics.mutual_info_score``
  (to 1e-13 abs).
- Subtracted Miller-Madow bias correction against its plain-Python definition
  (to 1e-13 abs), including exact null gating under non-unit weights.
- Exponentially decayed MI against longhand / ``scipy.stats.entropy`` (to 1e-12 abs).
- Windowed Spearman's rho against ``scipy.stats.spearmanr`` (to 1e-14 abs).
- Windowed Kendall's tau-b against ``scipy.stats.kendalltau(..., variant="b")``
  (to 1e-14 abs).
- Windowed Chatterjee's xi against ``scipy.stats.chatterjeexi`` on distinct ``x``
  (to 1e-14 abs) and against arrival-order tie longhand with ties in ``x``.
- ``ModelBank.dependence()`` parity with ``fit_predict`` / ``predict``, fresh
  reads under ``rank_every_rows > 1``, group filtering, readiness gating, and
  refusals.
- Chunk invariance, mid-stream save/load, zero-weight / null / NaN / inf rows,
  ``window_size`` (``"right"`` vs ``"both"``) with numeric and temporal clocks,
  ``bin_warm_span``, ``rank_every``, atomic ``max_state_bytes`` cap + half-cap
  warning, and spec refusals.
- 50-seed ``oracle_grid`` + ledger guard covering Gaussian MI convergence to
  Gauss-Legendre ``I_bin`` and exact parity against sklearn and scipy.
"""

from __future__ import annotations

import datetime
import math
import warnings

import numpy as np
import polars as pl
import pytest
from scipy import stats as sp_stats
from sklearn.metrics import mutual_info_score

import ledger
from rheon import polars as rp

TIER = "mixed"


# ---------------------------------------------------------------------------
# Plain-Python reference helpers
# ---------------------------------------------------------------------------


def _bin_index(edges: list[float], v: float) -> int:
    # Matches Rust `partition_point(|&e| e <= v)` (right-open bins, left-closed).
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if edges[mid] <= v:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _plain_mi_mm(
    xs: list[float],
    ys: list[float],
    ws: list[float],
    x_edges: list[float],
    y_edges: list[float],
) -> tuple[float, float]:
    bx = len(x_edges) + 1
    by = len(y_edges) + 1
    table = np.zeros((bx, by), dtype=np.float64)
    all_unit = True
    for x, y, w in zip(xs, ys, ws, strict=True):
        if w <= 0.0 or not (math.isfinite(x) and math.isfinite(y)):
            continue
        if w != 1.0:
            all_unit = False
        table[_bin_index(x_edges, x), _bin_index(y_edges, y)] += w
    n = float(table.sum())
    if n <= 0.0:
        return math.nan, math.nan
    na = table.sum(axis=1)
    nb = table.sum(axis=0)
    mi_sum = 0.0
    b_xy = 0
    for a in range(bx):
        if na[a] <= 0.0:
            continue
        for b in range(by):
            w_ab = float(table[a, b])
            if w_ab > 0.0:
                b_xy += 1
                mi_sum += w_ab * math.log((w_ab / float(na[a])) * (n / float(nb[b])))
    mi = max(0.0, mi_sum / n)
    if not all_unit:
        return mi, math.nan
    b_x = int(np.count_nonzero(na > 0.0))
    b_y = int(np.count_nonzero(nb > 0.0))
    mi_mm = mi - (b_xy - b_x - b_y + 1) / (2.0 * n)
    return mi, mi_mm


def _plain_xi_arrival_order(xs: list[float], ys: list[float]) -> float:
    """Chatterjee (2021) eq. 1.2 with deterministic arrival-order tie-breaking in x."""
    n = len(xs)
    if n < 2:
        return math.nan
    # Stable sort by x preserves arrival order among ties in x.
    order = sorted(range(n), key=lambda i: xs[i])
    y_sorted = [ys[i] for i in order]
    # r_i = #{j : y_j <= y_(i)}, l_i = #{j : y_j >= y_(i)}
    r = [sum(1 for v in y_sorted if v <= yv) for yv in y_sorted]
    lv = [sum(1 for v in y_sorted if v >= yv) for yv in y_sorted]
    denom = 2 * sum(v * (n - v) for v in lv)
    if denom == 0:
        return math.nan
    num = n * sum(abs(r[i + 1] - r[i]) for i in range(n - 1))
    return 1.0 - num / denom


def _gaussian_binned_mi(rho: float, x_edges: list[float], y_edges: list[float]) -> float:
    """Exact discrete MI of standard bivariate normal with correlation `rho`
    over rectangle cells defined by `x_edges` and `y_edges` via Gauss-Legendre
    quadrature."""
    # Map (-inf, inf) edges into probability space via Phi, integrate normal copula
    # density on each cell with 16-point Gauss-Legendre rule.
    nodes, weights = np.polynomial.legendre.leggauss(16)
    ux = [0.0, *(float(sp_stats.norm.cdf(e)) for e in x_edges), 1.0]
    uy = [0.0, *(float(sp_stats.norm.cdf(e)) for e in y_edges), 1.0]
    s = math.sqrt(1.0 - rho * rho)
    # Clip extreme tails at 1e-12 in u-space for finite ppf.
    eps = 1e-12
    bx, by = len(ux) - 1, len(uy) - 1
    p = np.zeros((bx, by), dtype=np.float64)
    for a in range(bx):
        u0, u1 = max(ux[a], eps), min(ux[a + 1], 1.0 - eps)
        us = 0.5 * (u1 - u0) * nodes + 0.5 * (u0 + u1)
        wu = 0.5 * (u1 - u0) * weights
        zs = sp_stats.norm.ppf(us)
        for b in range(by):
            # Conditional distribution: Y | X=z ~ N(rho * z, 1 - rho^2).
            y_lo = -math.inf if b == 0 else y_edges[b - 1]
            y_hi = math.inf if b == by - 1 else y_edges[b]
            cdf_hi = 1.0 if math.isinf(y_hi) else sp_stats.norm.cdf((y_hi - rho * zs) / s)
            cdf_lo = 0.0 if math.isinf(y_lo) else sp_stats.norm.cdf((y_lo - rho * zs) / s)
            p[a, b] = float(np.sum(wu * (cdf_hi - cdf_lo)))
    p /= p.sum()
    pa = p.sum(axis=1)
    pb = p.sum(axis=0)
    mi = 0.0
    for a in range(bx):
        for b in range(by):
            if p[a, b] > 0.0:
                mi += float(p[a, b]) * math.log(float(p[a, b]) / (float(pa[a]) * float(pb[b])))
    return max(0.0, mi)


# ---------------------------------------------------------------------------
# Exact oracle tests
# ---------------------------------------------------------------------------


def test_mi_and_miller_madow_match_sklearn_and_definition():
    rng = np.random.default_rng(101)
    n = 240
    x0 = rng.standard_normal(n)
    x1 = rng.uniform(-2.0, 2.0, n)
    y0 = x0**2 + 0.25 * rng.standard_normal(n)
    y1 = np.sin(2.0 * x1) + 0.1 * rng.standard_normal(n)
    xe = {"x0": [-1.0, -0.3, 0.3, 1.0], "x1": [-1.2, -0.4, 0.4, 1.2]}
    ye = {"y0": [0.2, 0.8, 1.6, 2.8], "y1": [-0.6, -0.1, 0.1, 0.6]}

    spec = rp.spec.dependence(
        "dep",
        targets=["y0", "y1"],
        features=["x0", "x1"],
        measures=["mi", "mi_mm"],
        bin_edges=xe,
        target_bin_edges=ye,
        window_rows=60,
    )
    df = pl.DataFrame({"x0": x0, "x1": x1, "y0": y0, "y1": y1})
    bank = rp.ModelBank([spec])
    out = bank.fit_predict(df).unnest("dep")

    for i in [1, 15, 59, 60, 120, 239]:
        start = max(0, i - 60)
        for f, xv in [("x0", x0), ("x1", x1)]:
            for t, yv in [("y0", y0), ("y1", y1)]:
                xb = [_bin_index(xe[f], float(v)) for v in xv[start:i]]
                yb = [_bin_index(ye[t], float(v)) for v in yv[start:i]]
                sk_mi = float(mutual_info_score(xb, yb))
                ref_mi, ref_mm = _plain_mi_mm(
                    xv[start:i].tolist(),
                    yv[start:i].tolist(),
                    [1.0] * (i - start),
                    xe[f],
                    ye[t],
                )
                assert abs(sk_mi - ref_mi) < 1e-13
                got_mi = out[f"mi_{f}_{t}"][i]
                got_mm = out[f"mi_mm_{f}_{t}"][i]
                assert got_mi is not None and abs(got_mi - sk_mi) < 1e-13
                assert got_mm is not None and abs(got_mm - ref_mm) < 1e-13
                # Subtracted Miller-Madow is <= plug-in MI whenever B_xy - B_x - B_y + 1 >= 0.
                assert got_mm <= got_mi + 1e-13

    # Verify table matches end-of-stream window [n - 60 : n]
    tbl = bank.dependence("dep")
    assert tbl.height == 4
    for row in tbl.iter_rows(named=True):
        f, t = row["feature"], row["target"]
        xv = x0 if f == "x0" else x1
        yv = y0 if t == "y0" else y1
        ref_mi, ref_mm = _plain_mi_mm(
            xv[n - 60 :].tolist(),
            yv[n - 60 :].tolist(),
            [1.0] * 60,
            xe[f],
            ye[t],
        )
        assert abs(row["mi"] - ref_mi) < 1e-13
        assert abs(row["mi_mm"] - ref_mm) < 1e-13


def test_decayed_mi_matches_scipy_entropy_and_longhand():
    rng = np.random.default_rng(202)
    n = 180
    hl = 15.0
    t_col = np.cumsum(rng.choice([0.5, 1.0, 2.0, 4.0], size=n))
    x = rng.standard_normal(n)
    y = 0.7 * x + 0.5 * rng.standard_normal(n)
    w = rng.uniform(0.25, 2.5, n)
    xe = [-1.0, 0.0, 1.0]
    ye = [-1.0, 0.0, 1.0]

    spec = rp.spec.dependence(
        "dep",
        targets=["y"],
        features=["x"],
        measures=["mi"],
        bin_edges=[xe],
        target_bin_edges=[ye],
        clock="t",
        gap_cap=10.0,
        half_life=hl,
        weight="w",
    )
    df = pl.DataFrame({"t": t_col, "x": x, "y": y, "w": w})
    bank = rp.ModelBank([spec])
    out = bank.fit_predict(df).unnest("dep")

    for i in [1, 10, 50, 120, 179]:
        # Effective decayed weights at time t_{i-1} (before row i is learned).
        ages = t_col[i - 1] - t_col[:i]
        dw = (w[:i] * (0.5 ** (ages / hl))).tolist()
        ref_mi, _ = _plain_mi_mm(x[:i].tolist(), y[:i].tolist(), dw, xe, ye)
        # Also verify against scipy.stats.entropy(p_x) + entropy(p_y) - entropy(p_xy).
        tbl = np.zeros((4, 4), dtype=np.float64)
        for r in range(i):
            tbl[_bin_index(xe, float(x[r])), _bin_index(ye, float(y[r]))] += dw[r]
        p = tbl / tbl.sum()
        hx = float(sp_stats.entropy(p.sum(axis=1)))
        hy = float(sp_stats.entropy(p.sum(axis=0)))
        hxy = float(sp_stats.entropy(p.ravel()))
        ent_mi = max(0.0, hx + hy - hxy)
        got = out["mi_x_y"][i]
        assert got is not None
        assert abs(got - ref_mi) < 1e-12
        assert abs(got - ent_mi) < 1e-12


def test_windowed_ranks_match_scipy_with_ties_and_missing_targets():
    rng = np.random.default_rng(303)
    n = 110
    w_rows = 30
    # x0: continuous (distinct, exact scipy chatterjeexi oracle);
    # x1: discrete with ties (arrival-order xi longhand + scipy spearman/kendall).
    x0 = rng.standard_normal(n)
    x1 = np.round(rng.uniform(-2.0, 2.0, n) * 2.0) / 2.0
    y0: list[float | None] = [
        None if i % 9 == 4 else float(x0[i] ** 2 + 0.2 * rng.standard_normal()) for i in range(n)
    ]
    y1: list[float | None] = [
        None if i % 11 == 5 else float(round(x1[i] + 0.5 * rng.standard_normal())) for i in range(n)
    ]

    spec = rp.spec.dependence(
        "dep",
        targets=["y0", "y1"],
        features=["x0", "x1"],
        measures=["spearman", "kendall", "xi"],
        window_rows=w_rows,
    )
    df = pl.DataFrame({"x0": x0, "x1": x1, "y0": y0, "y1": y1})
    bank = rp.ModelBank([spec])
    out = bank.fit_predict(df).unnest("dep")

    for i in range(n):
        start = max(0, i - w_rows)
        for f, xv in [("x0", x0), ("x1", x1)]:
            for t, yv in [("y0", y0), ("y1", y1)]:
                sx = [float(xv[r]) for r in range(start, i) if yv[r] is not None]
                sy = [float(yv[r]) for r in range(start, i) if yv[r] is not None]
                got_sp = out[f"spearman_{f}_{t}"][i]
                got_kd = out[f"kendall_{f}_{t}"][i]
                got_xi = out[f"xi_{f}_{t}"][i]
                if len(sx) < 2 or len(set(sx)) < 2 or len(set(sy)) < 2:
                    assert got_sp is None
                    assert got_kd is None
                else:
                    ref_sp = float(sp_stats.spearmanr(sx, sy).statistic)
                    ref_kd = float(sp_stats.kendalltau(sx, sy, variant="b").statistic)
                    assert got_sp is not None and abs(got_sp - ref_sp) < 1e-14
                    assert got_kd is not None and abs(got_kd - ref_kd) < 1e-14
                if len(sx) < 2 or len(set(sy)) < 2:
                    assert got_xi is None
                else:
                    ref_xi = _plain_xi_arrival_order(sx, sy)
                    assert got_xi is not None and abs(got_xi - ref_xi) < 1e-14
                    if f == "x0":
                        # Distinct x: matches scipy.stats.chatterjeexi exactly.
                        sc_xi = float(sp_stats.chatterjeexi(sx, sy).statistic)
                        assert abs(got_xi - sc_xi) < 1e-14


def test_miller_madow_null_under_non_unit_weight_and_recovers_after_window():
    rng = np.random.default_rng(404)
    n = 45
    x = rng.standard_normal(n)
    y = x + 0.2 * rng.standard_normal(n)
    w = [0.5 if i == 10 else 1.0 for i in range(n)]
    edges = [[-1.0, 0.0, 1.0]]

    spec = rp.spec.dependence(
        "dep",
        targets=["y"],
        features=["x"],
        measures=["mi", "mi_mm"],
        bin_edges=edges,
        target_bin_edges=edges,
        window_rows=12,
        weight="w",
    )
    out = rp.ModelBank([spec]).fit_predict(pl.DataFrame({"x": x, "y": y, "w": w})).unnest("dep")
    # Before row 10 is learned (rows 1..=10): unit weights -> mi_mm finite.
    assert out["mi_mm_x_y"][10] is not None
    # Rows 11..=22: row 10 (weight 0.5) is in the 12-row window -> mi_mm null, mi finite.
    for r in range(11, 23):
        assert out["mi_x_y"][r] is not None
        assert out["mi_mm_x_y"][r] is None
    # Row 23+: row 10 has exited -> unit weights restored -> mi_mm finite again.
    for r in range(23, n):
        assert out["mi_mm_x_y"][r] is not None


# ---------------------------------------------------------------------------
# ModelBank.dependence() table, cadence, gating, chunking, save/load
# ---------------------------------------------------------------------------


def test_bank_dependence_table_fresh_reads_groups_and_gating():
    rng = np.random.default_rng(505)
    n = 80
    df = pl.DataFrame(
        {
            "g": ["a" if i % 2 == 0 else "b" for i in range(n)],
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "y": rng.standard_normal(n),
        }
    )
    df = df.with_columns(y=pl.col("x0") ** 2 + 0.1 * pl.col("y"))

    spec = rp.spec.dependence(
        "dep",
        targets=["y"],
        features=["x0", "x1"],
        measures=["mi", "spearman", "kendall", "xi"],
        bins=6,
        bin_warm_rows=15,
        window_rows=25,
        rank_every_rows=10,
        group="g",
    )
    # Also test a second spec with rank_every_rows=1 to compare fresh table reads.
    spec_every = rp.spec.dependence(
        "dep_every",
        targets=["y"],
        features=["x0", "x1"],
        measures=["mi", "spearman", "kendall", "xi"],
        bins=6,
        bin_warm_rows=15,
        window_rows=25,
        rank_every_rows=1,
        group="g",
    )
    bank = rp.ModelBank([spec, spec_every])
    # Empty bank returns 0-row table with the exact schema.
    empty = bank.dependence("dep")
    assert empty.height == 0
    assert empty.columns == [
        "spec",
        "group",
        "instance",
        "half_life",
        "lam",
        "feature",
        "target",
        "weight_sum",
        "mi",
        "spearman",
        "kendall",
        "xi",
    ]

    bank.fit(df)
    t_throttled = bank.dependence("dep")
    t_fresh = bank.dependence("dep_every")
    # Bank.dependence computes fresh exact ranks even when rank_every_rows=10!
    assert t_throttled.drop("spec").equals(t_fresh.drop("spec"))
    assert t_throttled.height == 4
    assert sorted(t_throttled["group"].to_list()) == ["a", "a", "b", "b"]

    # Group filtering and unseen group key.
    only_a = bank.dependence("dep", group="a")
    assert only_a.height == 2
    assert only_a["group"].to_list() == ["a", "a"]
    assert bank.dependence("dep", group="missing").height == 0

    # Parity with bank.predict() on the final state (for rank_every_rows=1).
    probe = pl.DataFrame({"g": ["a", "b"], "x0": [0.0, 0.0], "x1": [0.0, 0.0], "y": [None, None]})
    pred = bank.predict(probe).unnest("dep_every")
    for gi, grp in enumerate(["a", "b"]):
        sub = t_fresh.filter(pl.col("group") == grp)
        for row in sub.iter_rows(named=True):
            f = row["feature"]
            for m in ("mi", "spearman", "kendall", "xi"):
                assert pred[f"{m}_{f}_y"][gi] == pytest.approx(row[m], abs=1e-14)

    # Readiness gating in bank.dependence(): min_weight > learned rows nulls measures.
    gated_spec = rp.spec.dependence(
        "gated",
        targets=["y"],
        features=["x0"],
        measures=["mi", "kendall"],
        bins=6,
        bin_warm_rows=10,
        window_rows=20,
        min_weight=50.0,
    )
    gb = rp.ModelBank([gated_spec])
    gb.fit(df.head(30))
    gt = gb.dependence("gated")
    assert gt["weight_sum"][0] == pytest.approx(20.0)
    assert gt["mi"][0] is None
    assert gt["kendall"][0] is None

    # Wrong spec kind refused clearly.
    other = rp.ModelBank([rp.spec.marginal("m", targets=["y"], features=["x0"], half_life=20.0)])
    with pytest.raises(ValueError, match='not "dependence"'):
        other.dependence("m")


def test_chunk_invariance_and_save_load_mid_stream():
    rng = np.random.default_rng(606)
    n = 120
    df = pl.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "y": rng.standard_normal(n),
            "w": [0.0 if i % 13 == 3 else 1.0 for i in range(n)],
        }
    )
    specs = [
        rp.spec.dependence(
            "dep_decay",
            targets=["y"],
            features=["x0", "x1"],
            measures=["mi"],
            bins=6,
            bin_warm_rows=18,
            clock="t",
            gap_cap=10.0,
            half_life=20.0,
            weight="w",
        ),
        rp.spec.dependence(
            "dep_rows",
            targets=["y"],
            features=["x0", "x1"],
            measures=["mi", "mi_mm", "spearman", "kendall", "xi"],
            bins=6,
            bin_warm_rows=15,
            window_rows=30,
            rank_every_rows=4,
            clock="t",
            gap_cap=10.0,
            weight="w",
        ),
        rp.spec.dependence(
            "dep_clock",
            targets=["y"],
            features=["x0"],
            measures=["mi", "spearman", "kendall", "xi"],
            bins=6,
            bin_warm_span=15.0,
            window_size=25.0,
            closed="both",
            rank_every=5.0,
            clock="t",
            gap_cap=10.0,
            weight="w",
        ),
    ]
    whole_bank = rp.ModelBank(specs)
    whole = whole_bank.fit_predict(df)

    # Chunked run with save/load mid-warmup (row 10) and post-warmup (row 55).
    chunk_bank = rp.ModelBank(specs)
    parts = [chunk_bank.fit_predict(df.slice(0, 10))]
    saved1 = chunk_bank.save_bytes()
    chunk_bank = rp.ModelBank.load_bytes(saved1, specs)
    parts.append(chunk_bank.fit_predict(df.slice(10, 45)))
    saved2 = chunk_bank.save_bytes()
    chunk_bank = rp.ModelBank.load_bytes(saved2, specs)
    parts.append(chunk_bank.fit_predict(df.slice(55, n - 55)))
    chunked = pl.concat(parts)

    assert whole.equals(chunked, null_equal=True)
    for s in specs:
        assert whole_bank.dependence(s["name"]).equals(
            chunk_bank.dependence(s["name"]), null_equal=True
        )


def test_clock_window_temporal_durations_closed_edges_and_held_cap():
    t0 = datetime.datetime(2026, 1, 1, 0, 0, 0)
    ts = [t0 + datetime.timedelta(seconds=i * 10) for i in range(40)]
    rng = np.random.default_rng(707)
    x = rng.standard_normal(40)
    y = x**2 + 0.1 * rng.standard_normal(40)
    df = pl.DataFrame({"t": ts, "x": x, "y": y})

    spec_r = rp.spec.dependence(
        "dep_r",
        targets=["y"],
        features=["x"],
        measures=["mi", "kendall", "spearman", "xi"],
        bins=5,
        bin_warm_span="50s",
        window_size="100s",
        closed="right",
        rank_every="20s",
        clock="t",
        gap_cap="60s",
    )
    spec_b = rp.spec.dependence(
        "dep_b",
        targets=["y"],
        features=["x"],
        measures=["mi", "kendall", "spearman", "xi"],
        bins=5,
        bin_warm_span="50s",
        window_size="100s",
        closed="both",
        rank_every="20s",
        clock="t",
        gap_cap="60s",
    )
    out = rp.ModelBank([spec_r, spec_b]).fit_predict(df)
    r_col = out["dep_r"].struct
    b_col = out["dep_b"].struct
    # At row 10 (t = 100s), row 0 (t = 0s) is exactly 100s old:
    # closed="right" drops row 0 (weight_sum = 9), closed="both" keeps it (weight_sum = 10).
    assert r_col.field("weight_sum")[10] == pytest.approx(9.0)
    assert b_col.field("weight_sum")[10] == pytest.approx(10.0)
    # Warming up sets withheld_reason = "below_warm_rows" on row 0..5.
    assert r_col.field("withheld_reason")[2] == "below_warm_rows"
    assert r_col.field("withheld_reason")[12] is None

    # HeldCap: half-cap notice raised once, refusal at 100% cap.
    dense_ts = [t0 + datetime.timedelta(milliseconds=i) for i in range(200)]
    dense = pl.DataFrame(
        {
            "t": dense_ts,
            "x": rng.standard_normal(200),
            "y": rng.standard_normal(200),
        }
    )
    capped = rp.spec.dependence(
        "cap",
        targets=["y"],
        features=["x"],
        measures=["kendall"],
        window_size="10s",
        clock="t",
        gap_cap="10s",
        max_state_bytes=4096,
    )
    bank = rp.ModelBank([capped])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bank.fit_predict(dense.head(40))
    assert any("past half of max_state_bytes" in str(w.message) for w in caught)
    with pytest.raises(ValueError, match="max_state_bytes"):
        bank.fit_predict(dense.slice(40))


def test_dependence_spec_refusals():
    base = dict(targets=["y"], features=["x"])
    with pytest.raises(ValueError, match="measures"):
        rp.spec.dependence("d", **base, measures=[])
    with pytest.raises(ValueError, match="measures"):
        rp.spec.dependence("d", **base, measures=["mi", "mi"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="measures"):
        rp.spec.dependence("d", **base, measures=["pearson"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="window_rows"):
        rp.spec.dependence("d", **base, measures=["spearman"])
    with pytest.raises(ValueError, match="decay"):
        rp.spec.dependence("d", **base, measures=["mi"], window_rows=50, half_life=20.0)
    with pytest.raises(ValueError, match="mi_mm"):
        rp.spec.dependence("d", **base, measures=["mi_mm"], half_life=20.0)
    with pytest.raises(ValueError, match="both"):
        rp.spec.dependence("d", **base, window_rows=20, window_size=20.0)
    with pytest.raises(ValueError, match="closed"):
        rp.spec.dependence("d", **base, window_rows=20, closed="both")
    with pytest.raises(ValueError, match="rank_every"):
        rp.spec.dependence("d", **base, measures=["mi"], rank_every_rows=5)
    with pytest.raises(ValueError, match="rank_every"):
        rp.spec.dependence("d", **base, measures=["kendall"], window_rows=20, rank_every_rows=5)
    with pytest.raises(ValueError, match="bin"):
        rp.spec.dependence("d", **base, measures=["kendall"], window_rows=20, bins=8)
    with pytest.raises(ValueError, match="bin_edges"):
        rp.spec.dependence("d", **base, bin_edges={"wrong": [0.0]})
    with pytest.raises(ValueError, match="target_bin_edges"):
        rp.spec.dependence("d", **base, target_bin_edges={"wrong": [0.0]})
    with pytest.raises(ValueError, match="emit_sigma"):
        rp.spec.dependence("d", **base, emit_sigma=True)


# ---------------------------------------------------------------------------
# 50-seed oracle grid + ledger guard
# ---------------------------------------------------------------------------

GRID_SEEDS = range(50)
# Standard-normal octile cut-points (exact to double precision, avoiding scipy in the hash guard).
_GAUSS_OCTILES = [
    -1.1503493803760079,
    -0.6744897501960817,
    -0.31863936396437514,
    0.0,
    0.31863936396437514,
    0.6744897501960817,
    1.1503493803760079,
]
_GRID_RHO = 0.8
_GRID_W = 200


def _grid_rheon(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    rng = np.random.default_rng(10_000 + seed)
    # 1) Bivariate Gaussian (rho = 0.8), 2,000 rows, fixed standard-normal quantile edges (8 bins)
    n_gauss = 2_000
    z1 = rng.standard_normal(n_gauss)
    z2 = _GRID_RHO * z1 + math.sqrt(1.0 - _GRID_RHO * _GRID_RHO) * rng.standard_normal(n_gauss)
    g_spec = rp.spec.dependence(
        "g",
        targets=["y"],
        features=["x"],
        measures=["mi", "mi_mm"],
        bin_edges=[_GAUSS_OCTILES],
        target_bin_edges=[_GAUSS_OCTILES],
    )
    g_bank = rp.ModelBank([g_spec])
    g_bank.fit(pl.DataFrame({"x": z1, "y": z2}))
    g_row = g_bank.dependence("g").row(0, named=True)

    # 2) Windowed nonlinear parabola (W = 200)
    n_rank = 300
    rx = rng.uniform(-2.0, 2.0, n_rank)
    ry = rx**2 + 0.15 * rng.standard_normal(n_rank)
    r_spec = rp.spec.dependence(
        "r",
        targets=["y"],
        features=["x"],
        measures=["spearman", "kendall", "xi"],
        window_rows=_GRID_W,
    )
    r_bank = rp.ModelBank([r_spec])
    r_bank.fit(pl.DataFrame({"x": rx, "y": ry}))
    r_row = r_bank.dependence("r").row(0, named=True)
    return z1, z2, rx, ry, g_row, r_row


def _grid_guard_outputs() -> list[list[float]]:
    out = []
    for seed in GRID_SEEDS:
        _, _, _, _, g_row, r_row = _grid_rheon(seed)
        out.append(
            [
                float(g_row["mi"]),
                float(g_row["mi_mm"]),
                float(r_row["spearman"]),
                float(r_row["kendall"]),
                float(r_row["xi"]),
            ]
        )
    return out


ledger.guard(
    "dependence",
    _grid_guard_outputs,
    oracles=["scipy", "scikit-learn"],
)


@pytest.mark.oracle_grid(
    reason="50 seeds of Gaussian MI convergence + sklearn/scipy rank oracle parity"
)
def test_fifty_seeds_dependence_oracles():
    i_bin = _gaussian_binned_mi(_GRID_RHO, _GAUSS_OCTILES, _GAUSS_OCTILES)
    mi_sklearn_gaps: list[float] = []
    mi_biases: list[float] = []
    mm_biases: list[float] = []
    sp_gaps: list[float] = []
    kd_gaps: list[float] = []
    xi_gaps: list[float] = []
    xi_parabolas: list[float] = []
    for seed in GRID_SEEDS:
        z1, z2, rx, ry, g_row, r_row = _grid_rheon(seed)
        xb = [_bin_index(_GAUSS_OCTILES, float(v)) for v in z1]
        yb = [_bin_index(_GAUSS_OCTILES, float(v)) for v in z2]
        sk_mi = float(mutual_info_score(xb, yb))
        mi_sklearn_gaps.append(abs(g_row["mi"] - sk_mi))
        mi_biases.append(g_row["mi"] - i_bin)
        mm_biases.append(g_row["mi_mm"] - i_bin)
        wx, wy = rx[-_GRID_W:], ry[-_GRID_W:]
        sp_gaps.append(abs(r_row["spearman"] - float(sp_stats.spearmanr(wx, wy).statistic)))
        kd_gaps.append(
            abs(r_row["kendall"] - float(sp_stats.kendalltau(wx, wy, variant="b").statistic))
        )
        xi_gaps.append(abs(r_row["xi"] - float(sp_stats.chatterjeexi(wx, wy).statistic)))
        xi_parabolas.append(float(r_row["xi"]))

    ledger.record("dependence", "mi_sklearn_gap", mi_sklearn_gaps, floor=1e-13)
    ledger.record("dependence", "mi_abs_err", [abs(b) for b in mi_biases])
    ledger.record("dependence", "mm_abs_err", [abs(b) for b in mm_biases])
    ledger.record("dependence", "spearman_gap", sp_gaps, floor=1e-14)
    ledger.record("dependence", "kendall_gap", kd_gaps, floor=1e-14)
    ledger.record("dependence", "xi_gap", xi_gaps, floor=1e-14)
    ledger.record("dependence", "xi_parabola", xi_parabolas, ideal=1.0)

    assert max(mi_sklearn_gaps) < 1e-13
    assert max(sp_gaps) < 1e-14
    assert max(kd_gaps) < 1e-14
    assert max(xi_gaps) < 1e-14
    # Miller-Madow cuts finite-sample positive MI bias on 8x8 Gaussian bins.
    assert abs(float(np.mean(mm_biases))) < abs(float(np.mean(mi_biases)))
    assert max(abs(b) for b in mm_biases) < 0.08
    assert min(xi_parabolas) > 0.70


def test_default_bin_warm_rows_scales_with_bins_and_cap():
    """Default MI bin_warm_rows derives from max(ceil(62.5 * max(Bx, By)), 5 * Bx * By)
    and clamps to fit max_state_bytes, while explicit overrides stay untouched."""
    import json

    from rheon import _rheon as _native
    from rheon import _spec

    def resolved_warm(**kw):
        spec = rp.spec.dependence("d", targets=["y"], features=["x"], **kw)
        return json.loads(_native.resolved_defaults(_spec._json(spec)))["derived"]["bin_warm_rows"]

    assert resolved_warm(bins=4, target_bins=4) == 250
    assert resolved_warm(bins=8, target_bins=8) == 500
    assert resolved_warm(bins=16, target_bins=16) == 1280
    assert resolved_warm(bins=4, target_bins=16) == 1000
    assert resolved_warm(bins=4, target_bins=4, bin_warm_rows=55) == 55

    # Tight max_state_bytes clamps default warm-up instead of refusing the spec.
    spec_cap = rp.spec.dependence(
        "d",
        targets=["y"],
        features=[f"x{i}" for i in range(10)],
        bins=32,
        target_bins=32,
        max_state_bytes=256 * 1024,
    )
    warm_cap = json.loads(_native.resolved_defaults(_spec._json(spec_cap)))["derived"][
        "bin_warm_rows"
    ]
    assert 2 <= warm_cap < 5120

    # End-to-end: bins=4 warms up at exactly 250 learned rows.
    rng = np.random.default_rng(123)
    x = rng.standard_normal(260)
    df = pl.DataFrame({"x": x, "y": x * x + 0.1 * rng.standard_normal(260)})
    s4 = rp.spec.dependence("d", targets=["y"], features=["x"], bins=4)
    out = rp.ModelBank([s4]).fit_predict(df)["d"].struct.unnest()
    assert out["mi_x_y"][249] is None
    assert out["mi_x_y"][250] is not None
