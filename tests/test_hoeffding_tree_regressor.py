"""Hoeffding tree regressor (FIMT with bounded QO): reference parity, ``river``
component & accuracy verification, memory-cap bounds, and bank contracts.

What is held, and why:

- **Exact parity against the independent longhand QO reference** (Mastelini &
  de Carvalho 2021, with both bugs in ``river`` 0.26.1's ``QOSplitter.__iter__``
  fixed: non-overlapping partition ``{slots <= i-1} | {slots >= i}`` and
  copied left-child statistics rather than aliasing ``aux_stats`` across
  candidates).
- **Direct component parity against ``river`` 0.26.1** on the variance
  reduction criterion and Hoeffding bound (``R = 1``).
- **50-seed oracle grid** (``@pytest.mark.oracle_grid``, registered in
  ``oracles.toml`` and ``oracle_hashes.toml``) against ``river`` 0.26.1's
  ``HoeffdingTreeRegressor`` with its unbounded ``TEBSTSplitter`` (using
  TE-BST as the accuracy benchmark because ``river``'s ``QOSplitter`` carries
  the two iterator bugs noted above), verifying bounded QO stays within a
  small relative MAE band of unbounded TE-BST across 50 seeds.
- **Multi-target, sparse nulls, formula targets, residual diagnostics,
  wide-feature ``ReadinessWarning``, zero-weight invariance, cap save/resume,
  chunk invariance, and refusals.**
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest

import ledger
from rheon import ReadinessWarning
from rheon import polars as rp

TIER = "mixed"

GRID_SEEDS = tuple(range(50))


# ---------------------------------------------------------------------------
# Independent longhand reference for bounded QO (Mastelini & de Carvalho 2021)


def _var_update(s: list[float], y: float, w: float) -> None:
    if w <= 0.0:
        return
    n_old, m_old, s_old = s
    n_new = n_old + w
    d1 = y - m_old
    m_new = m_old + (w / n_new) * d1
    d2 = y - m_new
    s[0] = n_new
    s[1] = m_new
    s[2] = s_old + w * d1 * d2


def _var_merge(a: list[float], b: list[float]) -> list[float]:
    if a[0] == 0.0:
        return list(b)
    if b[0] == 0.0:
        return list(a)
    n = a[0] + b[0]
    delta = b[1] - a[1]
    mean = a[1] + delta * (b[0] / n)
    ss = a[2] + b[2] + delta * delta * (a[0] * b[0] / n)
    return [n, mean, ss]


def _var_val(s: list[float]) -> float:
    return s[2] / (s[0] - 1.0) if s[0] > 1.0 else 0.0


def _vr(parent: list[float], left: list[float], right: list[float]) -> float:
    n = left[0] + right[0]
    if n <= 0.0:
        return 0.0
    return _var_val(parent) - (left[0] / n) * _var_val(left) - (right[0] / n) * _var_val(right)


class _RefQO:
    def __init__(self, radius: float, max_slots: int) -> None:
        self.radius = radius
        self.max_slots = max_slots
        self.slots: dict[int, list] = {}

    def update(self, x: float, y: float, w: float) -> None:
        idx = int(math.floor(x / self.radius))
        if idx in self.slots:
            slot = self.slots[idx]
            n_new = slot[0] + w
            slot[1] += (w / n_new) * (x - slot[1])
            slot[0] = n_new
            _var_update(slot[2], y, w)
        elif len(self.slots) < self.max_slots:
            self.slots[idx] = [w, x, [w, y, 0.0]]
        else:
            nearest = min(self.slots, key=lambda k: (abs(k - idx), k))
            slot = self.slots[nearest]
            n_new = slot[0] + w
            slot[1] += (w / n_new) * (x - slot[1])
            slot[0] = n_new
            _var_update(slot[2], y, w)

    def best_split(self, min_samples_split: float):
        if len(self.slots) < 2:
            return None
        keys = sorted(self.slots)
        suffix = [None] * len(keys)
        acc = [0.0, 0.0, 0.0]
        for i in range(len(keys) - 1, -1, -1):
            acc = _var_merge(self.slots[keys[i]][2], acc)
            suffix[i] = list(acc)
        pre = suffix[0]
        left = [0.0, 0.0, 0.0]
        best = None
        for i in range(1, len(keys)):
            left = _var_merge(left, self.slots[keys[i - 1]][2])
            right = suffix[i]
            merit = (
                _vr(pre, left, right)
                if (left[0] >= min_samples_split and right[0] >= min_samples_split)
                else 0.0
            )
            thr = 0.5 * (self.slots[keys[i - 1]][1] + self.slots[keys[i]][1])
            if best is None or merit > best[0]:
                best = (merit, thr, list(left), list(right))
        return best


class _RefHTRegressor:
    """Longhand FIMT-QO regressor matching the documented recursion."""

    def __init__(
        self,
        k: int,
        *,
        grace_period: float = 200.0,
        delta: float = 1e-7,
        tau: float = 0.05,
        radius: float = 0.25,
        max_slots: int = 32,
        min_samples_split: float = 5.0,
        max_depth: int = 980,
        merit_preprune: bool = True,
    ) -> None:
        self.k = k
        self.grace = grace_period
        self.delta = delta
        self.tau = tau
        self.radius = radius
        self.max_slots = max_slots
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.preprune = merit_preprune
        self.leaves: list[dict] = [
            {
                "stats": [0.0, 0.0, 0.0],
                "last": 0.0,
                "depth": 0,
                "active": max_depth > 0,
                "obs": None,
            }
        ]
        self.branches: list[list] = []
        self.root = ~0

    def predict(self, x: list[float]) -> float | None:
        node = self.root
        while node >= 0:
            f, t, le, ri = self.branches[node]
            node = le if x[f] <= t else ri
        leaf = self.leaves[~node]
        return leaf["stats"][1] if leaf["stats"][0] > 0.0 else None

    def learn(self, x: list[float], y: float, w: float) -> None:
        if w <= 0.0:
            return
        node, parent, right = self.root, -1, False
        while node >= 0:
            f, t, le, ri = self.branches[node]
            parent = node
            if x[f] <= t:
                node, right = le, False
            else:
                node, right = ri, True
        li = ~node
        leaf = self.leaves[li]
        _var_update(leaf["stats"], y, w)
        if not leaf["active"]:
            return
        if leaf["obs"] is None:
            leaf["obs"] = [_RefQO(self.radius, self.max_slots) for _ in range(self.k)]
        for f in range(self.k):
            leaf["obs"][f].update(x[f], y, w)
        if (
            leaf["stats"][0] - leaf["last"] >= self.grace
            and leaf["depth"] < self.max_depth
            and not self._attempt_split(li, parent, right)
        ):
            leaf["last"] = leaf["stats"][0]

    def _attempt_split(self, li: int, parent: int, right: bool) -> bool:
        leaf = self.leaves[li]
        if _var_val(leaf["stats"]) <= 0.0:
            return False
        cands = []
        for f in range(self.k):
            c = leaf["obs"][f].best_split(self.min_samples_split)
            if c is not None:
                cands.append((c[0], f, c[1], c[2], c[3]))
        if self.preprune:
            cands.append((0.0, None, 0.0, None, None))
        if not cands:
            return False
        cands.sort(key=lambda item: item[0])
        best = cands[-1]
        eps = math.sqrt(math.log(1.0 / self.delta) / (2.0 * leaf["stats"][0]))
        if len(cands) == 1:
            should = best[1] is not None and best[0] > 0.0
        else:
            second = cands[-2]
            if best[1] is None or best[0] <= 0.0:
                should = False
            else:
                ratio = second[0] / best[0]
                should = (ratio < 1.0 - eps) or (eps < self.tau)
        if not should:
            return False
        _, f, thr, ls, rs = best
        nd = leaf["depth"] + 1
        new_l = {
            "stats": ls,
            "last": ls[0],
            "depth": nd,
            "active": nd < self.max_depth,
            "obs": None,
        }
        new_r = {
            "stats": rs,
            "last": rs[0],
            "depth": nd,
            "active": nd < self.max_depth,
            "obs": None,
        }
        self.leaves[li] = new_l
        self.leaves.append(new_r)
        bid = len(self.branches)
        self.branches.append([f, thr, ~li, ~(len(self.leaves) - 1)])
        if parent < 0:
            self.root = bid
        elif right:
            self.branches[parent][3] = bid
        else:
            self.branches[parent][2] = bid
        return True


def _reg_stream(
    seed: int,
    *,
    n: int = 1200,
    k: int = 5,
    noise: float = 0.25,
    weighted: bool = False,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Nonlinear piecewise/Friedman-style regression stream."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k))
    y = (
        2.5 * (x[:, 0] > 0.0).astype(np.float64)
        - 1.8 * (x[:, 1] > 0.25).astype(np.float64)
        + 1.5 * ((x[:, 0] > 0.0) & (x[:, 2] < 0.0)).astype(np.float64)
        + 0.6 * x[:, 0] * x[:, 1]
        + noise * rng.standard_normal(n)
    )
    w = rng.uniform(0.5, 2.0, size=n) if weighted else np.ones(n, dtype=np.float64)
    cols: dict[str, object] = {f"x{j}": x[:, j] for j in range(k)}
    cols["y"] = y
    cols["w"] = w
    return pl.DataFrame(cols), x, y, w


def _rheon_reg_preds(df: pl.DataFrame, k: int, **kw: object) -> np.ndarray:
    spec = rp.spec.hoeffding_tree_regressor(
        "m",
        targets=["y"],
        features=[f"x{j}" for j in range(k)],
        **kw,
    )
    out = rp.ModelBank([spec]).fit_predict(df)["m"]
    return out.struct.field("pred_y").fill_null(np.nan).to_numpy()


def _run_river_tebst(
    x: np.ndarray,
    y: np.ndarray,
    *,
    grace_period: int = 50,
    delta: float = 1e-4,
    tau: float = 0.05,
) -> np.ndarray:
    from river.tree import HoeffdingTreeRegressor
    from river.tree.splitter import TEBSTSplitter

    tree = HoeffdingTreeRegressor(
        grace_period=grace_period,
        delta=delta,
        tau=tau,
        leaf_prediction="mean",
        splitter=TEBSTSplitter(),
        stop_mem_management=True,
    )
    n, k = x.shape
    preds = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        xi = {j: float(x[i, j]) for j in range(k)}
        if i > 0:
            preds[i] = float(tree.predict_one(xi))
        tree.learn_one(xi, float(y[i]))
    return preds


# ---------------------------------------------------------------------------
# Oracle-ledger guard (Rheon outputs only — never calls river)


def _reg_grid_outputs() -> list[np.ndarray]:
    out = []
    for s in GRID_SEEDS:
        df, _, _, _ = _reg_stream(s, n=1200, k=5, noise=0.25)
        preds = _rheon_reg_preds(df, 5, grace_period=50.0, delta=1e-4, tau=0.05)
        out.append(np.nan_to_num(preds[-400:], nan=-999.0))
    return out


ledger.guard("river_accuracy", _reg_grid_outputs, oracles=["river"])


# ---------------------------------------------------------------------------
# Reference parity and river component & accuracy checks


@pytest.mark.parametrize(
    ("weighted", "merit_preprune", "radius", "max_slots"),
    [
        (False, True, 0.25, 32),
        (True, True, 0.25, 32),
        (False, False, 0.5, 16),
    ],
)
def test_exact_parity_against_fixed_qo_reference(
    weighted: bool,
    merit_preprune: bool,
    radius: float,
    max_slots: int,
) -> None:
    df, x, y, w = _reg_stream(23, n=800, k=4, noise=0.2, weighted=weighted)
    kw: dict[str, object] = dict(
        grace_period=40.0,
        delta=1e-3,
        tau=0.05,
        radius=radius,
        max_slots=max_slots,
        merit_preprune=merit_preprune,
    )
    if weighted:
        kw["weight"] = "w"
    got = _rheon_reg_preds(df, 4, **kw)
    ref = _RefHTRegressor(
        4,
        grace_period=40.0,
        delta=1e-3,
        tau=0.05,
        radius=radius,
        max_slots=max_slots,
        merit_preprune=merit_preprune,
    )
    want = np.full(len(y), np.nan, dtype=np.float64)
    for i in range(len(y)):
        xi = x[i].tolist()
        p = ref.predict(xi)
        if p is not None:
            want[i] = p
        ref.learn(xi, float(y[i]), float(w[i]))
    assert np.isnan(got[0]) and np.isnan(want[0])
    np.testing.assert_allclose(got[1:], want[1:], rtol=1e-13, atol=1e-13)


def test_variance_reduction_and_hoeffding_bound_match_river_components() -> None:
    """Component-level parity against river's VarianceReductionSplitCriterion
    and Hoeffding bound."""
    from river.stats import Var
    from river.tree.split_criterion import VarianceReductionSplitCriterion

    crit = VarianceReductionSplitCriterion(min_samples_split=5)
    rng = np.random.default_rng(101)
    y_l = rng.normal(2.0, 0.5, size=60)
    y_r = rng.normal(-1.5, 0.7, size=80)
    vl, vr, vp = Var(), Var(), Var()
    sl, sr, sp = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    for v in y_l:
        vl.update(float(v))
        vp.update(float(v))
        _var_update(sl, float(v), 1.0)
        _var_update(sp, float(v), 1.0)
    for v in y_r:
        vr.update(float(v))
        vp.update(float(v))
        _var_update(sr, float(v), 1.0)
        _var_update(sp, float(v), 1.0)
    river_vr = crit.merit_of_split(vp, [vl, vr])
    our_vr = _vr(sp, sl, sr)
    assert math.isclose(our_vr, river_vr, rel_tol=1e-14, abs_tol=1e-14)
    assert crit.range_of_merit(vp) == 1.0


@pytest.mark.oracle_grid(reason="50-seed tail MAE comparison against river HoeffdingTreeRegressor")
def test_river_tebst_accuracy_50_seed_grid() -> None:
    """50-seed tail MAE comparison against river 0.26.1 HoeffdingTreeRegressor
    with unbounded TEBSTSplitter (used as benchmark because river's QOSplitter
    carries the two iterator bugs fixed in Rheon)."""
    rheon_maes = []
    river_maes = []
    excess_ratios = []
    for s in GRID_SEEDS:
        df, x, y, _ = _reg_stream(s, n=1200, k=5, noise=0.25)
        got = _rheon_reg_preds(df, 5, grace_period=50.0, delta=1e-4, tau=0.05)
        want = _run_river_tebst(x, y, grace_period=50, delta=1e-4, tau=0.05)
        tail_y = y[-400:]
        mae_rheon = float(np.mean(np.abs(tail_y - got[-400:])))
        mae_river = float(np.mean(np.abs(tail_y - want[-400:])))
        rheon_maes.append(mae_rheon)
        river_maes.append(mae_river)
        excess_ratios.append(max(0.0, mae_rheon / mae_river - 1.0))
    ledger.record("river_accuracy", "tail_mae_rheon", rheon_maes, ideal=0.0)
    ledger.record("river_accuracy", "tail_mae_excess_vs_tebst", excess_ratios, ideal=0.0)
    assert float(np.median(rheon_maes)) <= float(np.median(river_maes))
    assert float(np.median(excess_ratios)) < 0.05
    assert max(excess_ratios) < 0.30


# ---------------------------------------------------------------------------
# Multi-target, formula targets, diagnostics, ReadinessWarning, cap, contracts


def test_multitarget_sparse_nulls_and_residual_diagnostics() -> None:
    """Each target owns an independent tree: null on y1 teaches only y0,
    and residual diagnostics (sigma, conformal, metrics, drift) work."""
    df, x, y, _ = _reg_stream(7, n=600, k=3, noise=0.2)
    y2 = -2.0 * (x[:, 2] > 0.0).astype(np.float64) + 0.2 * np.random.default_rng(8).normal(
        size=len(y)
    )
    y2_col = [None if i % 5 == 0 else float(v) for i, v in enumerate(y2)]
    df2 = df.with_columns(pl.Series("y2", y2_col))
    spec_multi = rp.spec.hoeffding_tree_regressor(
        "m",
        targets=["y", "y2"],
        features=["x0", "x1", "x2"],
        grace_period=40.0,
        delta=1e-2,
        emit_sigma=True,
        conformal=0.9,
        emit_metrics=True,
        emit_drift=True,
    )
    spec_solo = rp.spec.hoeffding_tree_regressor(
        "s",
        targets=["y"],
        features=["x0", "x1", "x2"],
        grace_period=40.0,
        delta=1e-2,
    )
    out = rp.ModelBank([spec_multi, spec_solo]).fit_predict(df2)
    # Target y's predictions in the multi-target spec match the solo spec to the bit.
    assert out["m"].struct.field("pred_y").equals(out["s"].struct.field("pred_y"), null_equal=True)
    # Residual diagnostics populate cleanly.
    assert out["m"].struct.field("sigma_y").drop_nulls().len() > 500
    assert out["m"].struct.field("lo_y").drop_nulls().len() > 500
    assert out["m"].struct.field("r2_y").tail(1).item() > 0.4


def test_formula_target_matches_column_form() -> None:
    df, _, _, _ = _reg_stream(19, n=300, k=3, noise=0.2)
    df = df.with_columns(t=pl.arange(0, df.height).cast(pl.Float64))
    expr = rp.ops.ewm_mean("y", half_life=3.0, window_size=5.0, reverse=True).alias("fwd_y")
    spec_f = rp.spec.hoeffding_tree_regressor(
        "m",
        targets=[expr],
        features=["x0", "x1", "x2"],
        clock="t",
        gap_cap=10.0,
        embargo=6.0,
        grace_period=30.0,
        delta=1e-2,
    )
    col_df = rp.stream.with_windows(df, expr, like=spec_f)
    spec_c = rp.spec.hoeffding_tree_regressor(
        "m",
        targets=["fwd_y"],
        features=["x0", "x1", "x2"],
        clock="t",
        gap_cap=10.0,
        embargo=6.0,
        grace_period=30.0,
        delta=1e-2,
    )
    a = rp.ModelBank([spec_f]).fit_predict(df)["m"].struct.field("pred_fwd_y")
    b = rp.ModelBank([spec_c]).fit_predict(col_df)["m"].struct.field("pred_fwd_y")
    assert a.equals(b, null_equal=True)


def test_wide_regressor_readiness_warning() -> None:
    """When k >= 20 and max_state_bytes < 1 MiB, warn suggesting >= 1 MiB;
    when max_state_bytes >= 1 MiB (or k < 20), stay silent."""
    feats_20 = [f"x{i}" for i in range(20)]
    with pytest.warns(ReadinessWarning, match="1_048_576|1 MiB"):
        rp.spec.hoeffding_tree_regressor("m", targets=["y"], features=feats_20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rp.spec.hoeffding_tree_regressor(
            "m", targets=["y"], features=feats_20, max_state_bytes=1_048_576
        )
        rp.spec.hoeffding_tree_regressor("m", targets=["y"], features=feats_20[:19])
    assert not [w for w in caught if issubclass(w.category, ReadinessWarning)]


def test_zero_weight_invariance_and_cap_save_resume(tmp_path) -> None:
    df, _, _, _ = _reg_stream(41, n=2400, k=4, noise=0.2)
    # 1 active leaf reserve = 4 * (2 + 36 * 16) = 2312 bytes; cap at 6,000 bytes
    # allows ~2 active leaves at once and crosses the 1,000-row recheck twice.
    spec = rp.spec.hoeffding_tree_regressor(
        "m",
        targets=["y"],
        features=["x0", "x1", "x2", "x3"],
        weight="w",
        grace_period=40.0,
        delta=1e-2,
        max_slots=16,
        max_state_bytes=6000,
    )
    whole = rp.ModelBank([spec]).fit_predict(df)["m"]
    assert whole.struct.field("withheld_reason").drop_nulls().to_list() == []
    assert whole.struct.field("pred_y").null_count() == 1  # only row 0 before any target

    # Save/resume across recheck boundary is bit-identical.
    bank_a = rp.ModelBank([spec])
    p1 = bank_a.fit_predict(df.slice(0, 1234))["m"]
    path = tmp_path / "tree_reg.state"
    bank_a.save(path)
    p2 = rp.ModelBank.load(path).fit_predict(df.slice(1234))["m"]
    assert whole.equals(pl.concat([p1, p2]), null_equal=True)

    # Zero-weight rows with extreme values leave every weighted row bit-identical.
    rows = []
    for i in range(400):
        rows.append(df.row(i, named=True))
        if i % 20 == 7:
            rows.append({"x0": 1e6, "x1": -1e6, "x2": 1e6, "x3": -1e6, "y": 1e9, "w": 0.0})
    with_zeros = pl.DataFrame(rows)
    a = rp.ModelBank([spec]).fit_predict(df.head(400))["m"]
    b = rp.ModelBank([spec]).fit_predict(with_zeros).filter(pl.col("w") > 0.0)["m"]
    assert a.equals(b, null_equal=True)


def test_chunk_invariance() -> None:
    df, _, _, _ = _reg_stream(53, n=500, k=4, noise=0.2)
    spec = rp.spec.hoeffding_tree_regressor(
        "m", targets=["y"], features=["x0", "x1", "x2", "x3"], grace_period=30.0, delta=1e-2
    )
    one = rp.ModelBank([spec]).fit_predict(df)["m"]
    bank = rp.ModelBank([spec])
    many = pl.concat([bank.fit_predict(df.slice(i, 17))["m"] for i in range(0, df.height, 17)])
    assert one.equals(many, null_equal=True)


def test_regressor_refusals() -> None:
    base = dict(targets=["y"], features=["x0", "x1"])
    with pytest.raises(ValueError, match="half_life"):
        rp.spec.hoeffding_tree_regressor("m", **base, half_life=50.0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="lam"):
        rp.spec.hoeffding_tree_regressor("m", **base, lam=0.99)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="min_settled_frac"):
        rp.spec.hoeffding_tree_regressor("m", **base, min_settled_frac=0.5)
    with pytest.raises(ValueError, match="emit_se_coef"):
        rp.spec.hoeffding_tree_regressor("m", **base, emit_se_coef=True)
    with pytest.raises(ValueError, match="emit_influence"):
        rp.spec.hoeffding_tree_regressor("m", **base, emit_influence=True)
    with pytest.raises(ValueError, match="emit_robust_se"):
        rp.spec.hoeffding_tree_regressor("m", **base, emit_robust_se=True)
    with pytest.raises(ValueError, match="emit_error_inflation"):
        rp.spec.hoeffding_tree_regressor("m", **base, emit_error_inflation=True)
    with pytest.raises(ValueError, match="radius"):
        rp.spec.hoeffding_tree_regressor("m", **base, radius=0.0)
    with pytest.raises(ValueError, match="max_slots"):
        rp.spec.hoeffding_tree_regressor("m", **base, max_slots=1)
    with pytest.raises(ValueError, match="max_state_bytes"):
        rp.spec.hoeffding_tree_regressor("m", **base, max_state_bytes=100)
    with pytest.raises(ValueError, match="max_state_bytes"):
        rp.spec.hoeffding_tree_regressor("m", **base, max_state_bytes=64 * 1024 * 1024 + 1)
    # Multi-target splits the group cap equally across targets: cap that fits
    # 1 target (40 header + 35 leaf + 2 * (2 + 40 * 32) QO reserve = 2,639 B)
    # is too small for 2 targets.
    one_leaf_1t = 40 + 35 + 2 * (2 + 40 * 32)
    rp.spec.hoeffding_tree_regressor("m", **base, max_state_bytes=one_leaf_1t)
    with pytest.raises(ValueError, match="max_state_bytes"):
        rp.spec.hoeffding_tree_regressor(
            "m", targets=["y", "z"], features=["x0", "x1"], max_state_bytes=one_leaf_1t
        )
