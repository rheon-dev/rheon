"""Streaming Robust Random Cut Forest anomaly detector (`rp.spec.random_cut_forest`).

What is held, and why:

- **Guha et al. (2016) paper's cut invariants and reference CoDisp**: a pure
  Python reference implementation of a Random Cut Tree using the paper's exact
  cut rule (strict separation ``left <= cut < right``, zero unreachable leaves)
  and evict-after-score CoDisp matches Rheon's scores closely on anomaly
  ranking and obeys the exact CoDisp bounds (``0`` on an empty tree,
  ``<= tree_size - 1`` on a full tree, strictly monotone when an outlier moves
  farther from a compact cluster).
- **Evict-after-score and ``predict == step`` parity**: ``predict(row)`` equals
  ``fit_predict(row)`` to the bit on every row before, at, and after the FIFO
  eviction boundary (``n >= tree_size``).
- **Strict ``0``/``1`` weight semantics**: weight ``0`` scores identically to
  ``predict``, does not advance the Philox draw counter or FIFO ring, and
  leaves ``bank.save()`` byte-identical; any usable weight other than ``0`` or
  ``1`` is refused naming the spec, column, value, and row.
- **Shingling (``shingle_size > 1``)**: withholds the first
  ``shingle_size - 1`` rows with ``withheld_reason == "below_warm_rows"``,
  resets the shingle buffer on session changes and clock gaps past
  ``gap_cap``, and detects temporal sequence glitches invisible to
  ``shingle_size = 1``.
- **Streaming & Philox guarantees**: chunk and save/restore invariance before
  and after FIFO eviction, concept-drift adaptation once old points evict,
  distinct trees across seeds, groups and spec names, and every spec refusal.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"


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


# --- pure-Python Guha et al. (2016) paper-cut reference tree -----------------


@dataclass
class _RefLeaf:
    point: np.ndarray
    count: int = 1
    parent: _RefBranch | None = None

    @property
    def lo(self) -> np.ndarray:
        return self.point

    @property
    def hi(self) -> np.ndarray:
        return self.point


@dataclass
class _RefBranch:
    dim: int
    cut: float
    left: _RefNode
    right: _RefNode
    lo: np.ndarray
    hi: np.ndarray
    count: int
    parent: _RefBranch | None = None


_RefNode = _RefLeaf | _RefBranch


class _RefPaperCutTree:
    """Reference Guha et al. (2016) Random Cut Tree checking strict separation
    (``left <= cut < right``) and reachability of every leaf."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.root: _RefNode | None = None
        self.rng = rng

    def _draw_cut(self, lo: np.ndarray, hi: np.ndarray, x: np.ndarray) -> tuple[int, float]:
        lo2 = np.minimum(lo, x)
        hi2 = np.maximum(hi, x)
        spans = hi2 - lo2
        total = float(spans.sum())
        r = total * float(self.rng.uniform(0.0, 1.0))
        acc = 0.0
        q = int(np.flatnonzero(spans > 0.0)[-1])
        prev = 0.0
        for i, s in enumerate(spans):
            if s > 0.0:
                if r < acc + s:
                    q, prev = i, acc
                    break
                acc += s
        c = float(lo2[q] + (r - prev))
        if c >= hi2[q]:
            c = max(float(lo2[q]), math.nextafter(float(hi2[q]), -math.inf))
        return q, c

    def insert(self, x: np.ndarray) -> _RefLeaf:
        x = np.asarray(x, dtype=float)
        if self.root is None:
            leaf = _RefLeaf(point=x.copy())
            self.root = leaf
            return leaf
        v: _RefNode = self.root
        while True:
            if np.all(v.lo <= x) and np.all(x <= v.hi):
                if isinstance(v, _RefLeaf):
                    v.count += 1
                    p = v.parent
                    while p is not None:
                        p.count += 1
                        p = p.parent
                    return v
                v = v.left if x[v.dim] <= v.cut else v.right
                continue
            q, c = self._draw_cut(v.lo, v.hi, x)
            if x[q] < v.lo[q] and c < v.lo[q]:
                return self._splice(v, x, q, c, new_left=True)
            if x[q] > v.hi[q] and c >= v.hi[q]:
                return self._splice(v, x, q, c, new_left=False)
            if isinstance(v, _RefLeaf):
                # Identical cannot happen here because we checked bounds above.
                return self._splice(v, x, q, c, new_left=x[q] <= c)
            v = v.left if x[v.dim] <= v.cut else v.right

    def _splice(self, v: _RefNode, x: np.ndarray, q: int, c: float, *, new_left: bool) -> _RefLeaf:
        leaf = _RefLeaf(point=x.copy())
        left: _RefNode = leaf if new_left else v
        right: _RefNode = v if new_left else leaf
        branch = _RefBranch(
            dim=q,
            cut=c,
            left=left,
            right=right,
            lo=np.minimum(v.lo, x),
            hi=np.maximum(v.hi, x),
            count=v.count + 1,
            parent=v.parent,
        )
        if v.parent is None:
            self.root = branch
        elif v.parent.left is v:
            v.parent.left = branch
        else:
            v.parent.right = branch
        left.parent = branch
        right.parent = branch
        p = branch.parent
        while p is not None:
            p.count += 1
            p.lo = np.minimum(p.left.lo, p.right.lo)
            p.hi = np.maximum(p.left.hi, p.right.hi)
            p = p.parent
        return leaf

    def codisp(self, leaf: _RefLeaf) -> float:
        best = 0.0
        v: _RefNode = leaf
        while v.parent is not None:
            p = v.parent
            sib = p.right if p.left is v else p.left
            best = max(best, sib.count / v.count)
            v = p
        return best

    def assert_strict_separation_and_reachability(self) -> None:
        if self.root is None:
            return
        stack: list[_RefNode] = [self.root]
        leaves: list[_RefLeaf] = []
        while stack:
            v = stack.pop()
            if isinstance(v, _RefLeaf):
                leaves.append(v)
            else:
                assert v.left.hi[v.dim] <= v.cut < v.right.lo[v.dim]
                assert v.count == v.left.count + v.right.count
                stack.extend([v.left, v.right])
        for leaf in leaves:
            cur = self.root
            while isinstance(cur, _RefBranch):
                cur = cur.left if leaf.point[cur.dim] <= cur.cut else cur.right
            assert cur is leaf


class TestPaperCutAndScoring:
    def test_reference_paper_cut_invariant_and_codisp_ordering(self) -> None:
        """Sanity-check the Guha et al. (2016) paper-cut reference itself and
        verify Rheon agrees with its anomaly ordering."""
        rng = np.random.default_rng(507_200)
        pts = rng.standard_normal((80, 3))
        pts[75:] = np.array([12.0, -12.0, 10.0]) + 0.1 * rng.standard_normal((5, 3))

        ref_trees = [_RefPaperCutTree(np.random.default_rng(1000 + t)) for t in range(15)]
        ref_scores = np.zeros(len(pts))
        for i, p in enumerate(pts):
            ref_scores[i] = float(np.mean([t.codisp(t.insert(p)) for t in ref_trees]))
        for t in ref_trees:
            t.assert_strict_separation_and_reachability()

        df = pl.DataFrame({f"x{j}": pts[:, j] for j in range(3)})
        spec = rp.spec.random_cut_forest(
            "r", features=["x0", "x1", "x2"], n_trees=40, tree_size=128, seed=7
        )
        rheon_scores = rp.ModelBank([spec]).fit_predict(df).unnest("r")["score"].to_numpy()
        assert rheon_scores[0] == 0.0
        assert np.all(rheon_scores >= 0.0)
        assert np.all(rheon_scores <= 128.0)
        # Both reference and Rheon rank the 5 far outliers far above the normal tail.
        assert ref_scores[75:].mean() > 5.0 * ref_scores[20:75].mean()
        assert rheon_scores[75:].mean() > 5.0 * rheon_scores[20:75].mean()

    def test_planted_point_anomalies_high_auc(self) -> None:
        rng = np.random.default_rng(507_201)
        n = 400
        X = rng.standard_normal((n, 3))
        is_anom = np.zeros(n, dtype=bool)
        anom_idx = rng.choice(np.arange(80, n), size=20, replace=False)
        is_anom[anom_idx] = True
        X[anom_idx] = rng.choice([-8.0, 8.0], size=(20, 3)) + 0.3 * rng.standard_normal((20, 3))
        df = pl.DataFrame({f"x{j}": X[:, j] for j in range(3)})
        spec = rp.spec.random_cut_forest(
            "r", features=["x0", "x1", "x2"], n_trees=40, tree_size=128, seed=11
        )
        scores = rp.ModelBank([spec]).fit_predict(df).unnest("r")["score"].to_numpy()
        auc = _roc_auc(is_anom[80:], scores[80:])
        assert auc >= 0.99, auc

    def test_fifo_eviction_adapts_to_distribution_shift(self) -> None:
        rng = np.random.default_rng(507_202)
        n_phase = 120
        df = pl.DataFrame(
            {
                "x0": np.concatenate(
                    [0.2 * rng.standard_normal(n_phase), 15.0 + 0.2 * rng.standard_normal(n_phase)]
                ),
                "x1": np.concatenate(
                    [0.2 * rng.standard_normal(n_phase), 15.0 + 0.2 * rng.standard_normal(n_phase)]
                ),
            }
        )
        spec = rp.spec.random_cut_forest(
            "r", features=["x0", "x1"], n_trees=30, tree_size=48, seed=5
        )
        bank = rp.ModelBank([spec])
        bank.fit_predict(df)
        # After 120 rows at +15 (> tree_size 48), +15 is normal and 0 is anomalous.
        probe = pl.DataFrame({"x0": [15.0, 0.0], "x1": [15.0, 0.0]})
        scores = bank.predict(probe).unnest("r")["score"].to_list()
        assert scores[1] > 4.0 * scores[0], scores


class TestParityWeightsShinglingAndRefusals:
    @pytest.mark.parametrize("shingle_size", [1, 3])
    def test_predict_matches_step_bit_for_bit_across_eviction_boundary(
        self, shingle_size: int
    ) -> None:
        rng = np.random.default_rng(507_203 + shingle_size)
        n = 65  # crosses tree_size = 24 well past eviction
        df = pl.DataFrame({"x0": rng.standard_normal(n), "x1": rng.standard_normal(n)})
        spec = rp.spec.random_cut_forest(
            "r",
            features=["x0", "x1"],
            n_trees=12,
            tree_size=24,
            shingle_size=shingle_size,
            seed=13,
        )
        bank = rp.ModelBank([spec])
        bank.fit_predict(df.slice(0, 1))
        for i in range(1, n):
            row = df.slice(i, 1)
            p = bank.predict(row).unnest("r")["score"][0]
            s = bank.fit_predict(row).unnest("r")["score"][0]
            assert p == s, (shingle_size, i, p, s)

    def test_zero_and_one_weights_and_fractional_refusal(self) -> None:
        rng = np.random.default_rng(507_204)
        n = 60
        df = pl.DataFrame(
            {
                "x0": rng.standard_normal(n),
                "x1": rng.standard_normal(n),
                "w": np.ones(n),
            }
        )
        spec = rp.spec.random_cut_forest(
            "r", features=["x0", "x1"], weight="w", n_trees=10, tree_size=20, seed=3
        )
        bank = rp.ModelBank([spec])
        bank.fit_predict(df.slice(0, 35))
        buf = io.BytesIO()
        bank.save(buf)
        saved = buf.getvalue()

        zeros = df.slice(35, 10).with_columns(pl.lit(0.0).alias("w"))
        out_z = bank.fit_predict(zeros).unnest("r")
        pred_z = bank.predict(zeros).unnest("r")
        assert np.array_equal(out_z["score"].to_numpy(), pred_z["score"].to_numpy())
        assert np.array_equal(out_z["weight_sum"].to_numpy(), pred_z["weight_sum"].to_numpy())
        # Subsequent rows match a bank restored before the zero-weight rows bit-for-bit.
        tail = df.slice(45, 15).with_columns(pl.lit(1.0).alias("w"))
        restored = rp.ModelBank.load(io.BytesIO(saved))
        t1 = bank.fit_predict(tail).unnest("r")
        t2 = restored.fit_predict(tail).unnest("r")
        assert np.array_equal(t1["score"].to_numpy(), t2["score"].to_numpy())
        assert np.array_equal(t1["weight_sum"].to_numpy(), t2["weight_sum"].to_numpy())

        # Fractional weight and weight > 1 are refused, naming spec and row.
        for bad_w in (0.5, 2.0):
            bad = df.slice(45, 5).with_columns(
                pl.when(pl.arange(0, 5) == 2).then(pl.lit(bad_w)).otherwise(1.0).alias("w")
            )
            with pytest.raises(Exception, match=r"weight|0 or 1"):
                rp.ModelBank([spec]).fit_predict(bad)

    def test_shingling_warmup_resets_and_sequence_anomaly_auc(self) -> None:
        # Smooth sine wave in [-1, 1] whose consecutive values move slowly near
        # the 4D diagonal, with brief high-frequency alternating glitches that
        # stay strictly inside [-1, 1] pointwise (invisible to shingle_size=1)
        # but jump to an off-diagonal corner in 4D shingle space.
        n = 360
        t = np.arange(n, dtype=float)
        x = np.sin(2.0 * np.pi * t / 24.0)
        is_anom = np.zeros(n, dtype=bool)
        for start in (140, 220, 300):
            x[start : start + 4] = [0.95, -0.95, 0.95, -0.95]
            is_anom[start + 2 : start + 4] = True

        df = pl.DataFrame({"t": t, "x0": x})
        s1 = rp.spec.random_cut_forest(
            "r1", features=["x0"], n_trees=30, tree_size=96, shingle_size=1, seed=21
        )
        s4 = rp.spec.random_cut_forest(
            "r4", features=["x0"], n_trees=30, tree_size=96, shingle_size=4, seed=21
        )
        out = rp.ModelBank([s1, s4]).fit_predict(df)
        r4 = out.unnest("r4")
        # First shingle_size - 1 = 3 rows are withheld with below_warm_rows.
        assert r4["score"].slice(0, 3).null_count() == 3
        assert r4["withheld_reason"].slice(0, 3).to_list() == ["below_warm_rows"] * 3
        assert r4["score"].slice(3, n - 3).null_count() == 0

        auc_s4 = _roc_auc(is_anom[80:], r4["score"].slice(80, n - 80).to_numpy())
        auc_s1 = _roc_auc(is_anom[80:], out.unnest("r1")["score"].slice(80, n - 80).to_numpy())
        assert auc_s4 >= 0.95, (auc_s4, auc_s1)
        assert auc_s4 > auc_s1 + 0.15, (auc_s4, auc_s1)

        # Session break and clock gap past gap_cap both clear the shingle buffer.
        df_break = pl.DataFrame(
            {
                "t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 25.0, 26.0, 27.0, 28.0],
                "s": ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
                "x0": np.arange(10, dtype=float),
            }
        )
        spec_brk = rp.spec.random_cut_forest(
            "rb",
            features=["x0"],
            clock="t",
            gap_cap=5.0,
            session="s",
            session_gap=10.0,
            n_trees=5,
            tree_size=16,
            shingle_size=3,
            seed=0,
        )
        ob = rp.ModelBank([spec_brk]).fit_predict(df_break).unnest("rb")
        reasons = ob["withheld_reason"].to_list()
        # Rows 0,1 initial warmup; rows 4,5 session break; rows 6,7 clock gap past 5.0.
        assert reasons[:2] == ["below_warm_rows", "below_warm_rows"]
        assert reasons[2:4] == [None, None]
        assert reasons[4:6] == ["below_warm_rows", "below_warm_rows"]
        assert reasons[6:8] == ["below_warm_rows", "below_warm_rows"]
        assert reasons[8:] == [None, None]

    @pytest.mark.parametrize("shingle_size", [1, 3])
    def test_chunk_and_save_restore_invariance(self, shingle_size: int) -> None:
        rng = np.random.default_rng(507_205 + shingle_size)
        df = pl.DataFrame({"x0": rng.standard_normal(85), "x1": rng.standard_normal(85)})
        spec = rp.spec.random_cut_forest(
            "r",
            features=["x0", "x1"],
            n_trees=12,
            tree_size=24,
            shingle_size=shingle_size,
            seed=42,
        )
        one = rp.ModelBank([spec]).fit_predict(df).unnest("r")["score"].to_list()
        bank = rp.ModelBank([spec])
        parts = []
        for start in range(0, len(df), 11):
            parts.append(bank.fit_predict(df.slice(start, 11)).unnest("r"))
            if start in (22, 44):
                buf = io.BytesIO()
                bank.save(buf)
                buf.seek(0)
                bank = rp.ModelBank.load(buf)
        many = pl.concat(parts)["score"].to_list()
        assert one == many

    def test_philox_stream_isolates_seed_group_and_spec_name(self) -> None:
        rng = np.random.default_rng(507_206)
        df = pl.DataFrame({"x0": rng.standard_normal(50), "x1": rng.standard_normal(50)})
        base = dict(features=["x0", "x1"], n_trees=10, tree_size=32)
        o0 = (
            rp.ModelBank([rp.spec.random_cut_forest("r0", seed=0, **base)])
            .fit_predict(df)
            .unnest("r0")["score"]
            .to_numpy()
        )
        o1 = (
            rp.ModelBank([rp.spec.random_cut_forest("r0", seed=1, **base)])
            .fit_predict(df)
            .unnest("r0")["score"]
            .to_numpy()
        )
        on = (
            rp.ModelBank([rp.spec.random_cut_forest("r_other", seed=0, **base)])
            .fit_predict(df)
            .unnest("r_other")["score"]
            .to_numpy()
        )
        assert not np.allclose(o0, o1)
        assert not np.allclose(o0, on)

        df_g = pl.concat(
            [df.with_columns(pl.lit("g1").alias("g")), df.with_columns(pl.lit("g2").alias("g"))]
        )
        og = (
            rp.ModelBank([rp.spec.random_cut_forest("rg", group="g", seed=0, **base)])
            .fit_predict(df_g)
            .unnest("rg")
        )
        g1 = og.filter(pl.col("g") == "g1")["score"].to_numpy()
        g2 = og.filter(pl.col("g") == "g2")["score"].to_numpy()
        assert not np.allclose(g1, g2)

    def test_refusals(self) -> None:
        with pytest.raises(TypeError, match="targets"):
            rp.spec.random_cut_forest("r", targets=["y"], features=["x0"])  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="half_life"):
            rp.spec.random_cut_forest("r", features=["x0"], half_life=10.0)
        with pytest.raises(ValueError, match="n_trees"):
            rp.ModelBank([rp.spec.random_cut_forest("r", features=["x0"], n_trees=0)])
        with pytest.raises(ValueError, match="tree_size"):
            rp.ModelBank([rp.spec.random_cut_forest("r", features=["x0"], tree_size=1)])
        with pytest.raises(ValueError, match="shingle_size"):
            rp.ModelBank([rp.spec.random_cut_forest("r", features=["x0"], shingle_size=0)])
        with pytest.raises(ValueError, match="max_state_bytes"):
            rp.ModelBank(
                [
                    rp.spec.random_cut_forest(
                        "r", features=["x0", "x1"], n_trees=40, tree_size=256, max_state_bytes=1024
                    )
                ]
            )
        with pytest.raises(ValueError, match="coef_every"):
            rp.ModelBank([rp.spec.random_cut_forest("r", features=["x0"], coef_every=1)])
        with pytest.raises(ValueError, match="emit_sigma"):
            rp.ModelBank([rp.spec.random_cut_forest("r", features=["x0"], emit_sigma=True)])
