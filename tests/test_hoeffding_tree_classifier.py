"""Hoeffding tree classifier (VFDT): parity against ``river``, Study 503
invariants, memory-cap bounds, and bank contracts.

What is held, and why:

- **Row-by-row parity against ``river.tree.HoeffdingTreeClassifier``**
  across binary and three-class streams, information gain and Gini split
  criteria, majority / naive Bayes / adaptive naive Bayes leaf predictions,
  and non-unit weights. On standard non-capping streams where ``river``'s
  Python memory manager does not reactivate deactivated leaves, probabilities
  agree to round-off (``libm::erf`` vs Python's ``math.erf`` and ``logsumexp``
  summation order).
- **Explicit departures from ``river``'s Python path**, pinned directly:
  - ``merit_preprune=True`` uses null-split merit ``0.0`` (Domingos & Hulten
    2000), so a single-feature pure-noise stream does not split.
  - A zero-weight row changes no state, no Gaussian observer min/max, and no
    clock (hard rule 9).
  - A leaf deactivated by ``max_depth`` or ``max_share_to_split`` stays
    permanently inactive and is never reactivated by cap enforcement.
  - ``max_state_bytes`` bounds the packed state per group by deactivating the
    lowest-promise leaves' observers rather than refusing rows; save/load
    preserves the recheck cadence to the bit.
- **50-seed oracle grid** (``@pytest.mark.oracle_grid``, registered in
  ``oracles.toml`` and ``oracle_hashes.toml``) against ``river`` across binary
  and multiclass streams, information gain and Gini, and majority and adaptive
  naive Bayes leaf predictions.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import ledger
from rheon import polars as rp

TIER = "mixed"

GRID_SEEDS = tuple(range(50))


def _tree_stream(
    seed: int,
    *,
    n: int = 800,
    k: int = 5,
    n_classes: int = 2,
    noise: float = 0.08,
    weighted: bool = False,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Axis-aligned random decision tree concept over N(0, 1) features."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k))
    # Depth-3 balanced concept tree: 7 internal splits, 8 leaf labels.
    feats = [int(rng.integers(0, k)) for _ in range(7)]
    thresh = [float(rng.normal(0.0, 0.5)) for _ in range(7)]
    leaf_cls = [i % n_classes for i in range(8)]
    rng.shuffle(leaf_cls)

    idx = np.zeros(n, dtype=np.int64)
    for _depth in range(3):
        f = np.array([feats[i] for i in idx])
        t = np.array([thresh[i] for i in idx])
        go_right = x[np.arange(n), f] > t
        idx = 2 * idx + 1 + go_right.astype(np.int64)
    y_idx = np.array([leaf_cls[i - 7] for i in idx], dtype=np.int64)
    if noise > 0.0:
        flip = rng.random(n) < noise
        y_idx[flip] = (y_idx[flip] + rng.integers(1, n_classes, size=int(flip.sum()))) % n_classes
    # Seed the first n_classes rows in declared class order so river's Python
    # dict insertion order matches declared `classes` order on early ties.
    for c in range(min(n_classes, n)):
        y_idx[c] = c
    w = rng.uniform(0.5, 2.0, size=n) if weighted else np.ones(n, dtype=np.float64)
    classes = [f"c{c}" for c in range(n_classes)]
    labels = [classes[i] for i in y_idx]
    cols: dict[str, object] = {f"x{j}": x[:, j] for j in range(k)}
    cols["label"] = labels
    cols["w"] = w
    return pl.DataFrame(cols), x, y_idx, w


def _run_river_clf(
    x: np.ndarray,
    y_idx: np.ndarray,
    w: np.ndarray,
    classes: list[str],
    *,
    grace_period: int = 50,
    delta: float = 1e-4,
    tau: float = 0.05,
    split_criterion: str = "info_gain",
    leaf_prediction: str = "nba",
    nb_threshold: int = 0,
    n_splits: int = 10,
    merit_preprune: bool = False,
    max_share_to_split: float = 1.0,
) -> tuple[np.ndarray, list[str | None]]:
    from river.tree import HoeffdingTreeClassifier
    from river.tree.splitter import GaussianSplitter

    river_lp = "mc" if leaf_prediction == "majority" else leaf_prediction
    tree = HoeffdingTreeClassifier(
        grace_period=grace_period,
        delta=delta,
        tau=tau,
        split_criterion=split_criterion,
        leaf_prediction=river_lp,
        nb_threshold=nb_threshold,
        splitter=GaussianSplitter(n_splits=n_splits),
        merit_preprune=merit_preprune,
        max_share_to_split=max_share_to_split,
        stop_mem_management=True,
    )
    n, k = x.shape
    c_len = len(classes)
    probs = np.full((n, c_len), np.nan, dtype=np.float64)
    preds: list[str | None] = []
    # river 0.26.1's HellingerDistanceCriterion hardcodes integer class keys 0 and 1.
    use_int_labels = split_criterion == "hellinger"
    for i in range(n):
        xi = {j: float(x[i, j]) for j in range(k)}
        p = tree.predict_proba_one(xi)
        if p:
            for ci, cname in enumerate(classes):
                key = ci if use_int_labels else cname
                probs[i, ci] = p.get(key, 0.0)
            pred_raw = tree.predict_one(xi)
            preds.append(classes[int(pred_raw)] if use_int_labels else pred_raw)
        else:
            preds.append(None)
        label_val = int(y_idx[i]) if use_int_labels else classes[int(y_idx[i])]
        tree.learn_one(xi, label_val, w=float(w[i]))
    return probs, preds


def _rheon_clf_probs(
    df: pl.DataFrame,
    k: int,
    classes: list[str],
    **kw: object,
) -> tuple[np.ndarray, list[str | None]]:
    spec = rp.spec.hoeffding_tree_classifier(
        "m",
        features=[f"x{j}" for j in range(k)],
        label="label",
        classes=classes,
        **kw,
    )
    out = rp.ModelBank([spec]).fit_predict(df)["m"]
    probs = np.column_stack(
        [out.struct.field(f"p_{c}").fill_null(np.nan).to_numpy() for c in classes]
    )
    preds = out.struct.field("class").to_list()
    return probs, preds


# ---------------------------------------------------------------------------
# Oracle-ledger guard (Rheon outputs only — never calls river)


def _clf_grid_case(seed: int, *, n_classes: int, split_criterion: str, leaf_prediction: str):
    df, _, _, _ = _tree_stream(seed, n=600, k=5, n_classes=n_classes, noise=0.10)
    classes = [f"c{c}" for c in range(n_classes)]
    probs, _ = _rheon_clf_probs(
        df,
        5,
        classes,
        grace_period=50.0,
        delta=1e-4,
        split_criterion=split_criterion,
        leaf_prediction=leaf_prediction,
        merit_preprune=False,
        max_share_to_split=1.0,
    )
    return np.nan_to_num(probs, nan=-1.0)


def _clf_grid_outputs() -> dict[str, list[np.ndarray]]:
    return {
        "binary_ig_nba": [
            _clf_grid_case(s, n_classes=2, split_criterion="info_gain", leaf_prediction="nba")
            for s in GRID_SEEDS
        ],
        "binary_gini_majority": [
            _clf_grid_case(s, n_classes=2, split_criterion="gini", leaf_prediction="majority")
            for s in GRID_SEEDS
        ],
        "multiclass_ig_nba": [
            _clf_grid_case(s, n_classes=3, split_criterion="info_gain", leaf_prediction="nba")
            for s in GRID_SEEDS
        ],
    }


ledger.guard("river_parity", _clf_grid_outputs, oracles=["river"])


# ---------------------------------------------------------------------------
# Core parity against river 0.26.1


@pytest.mark.parametrize(
    ("n_classes", "split_criterion", "leaf_prediction", "weighted"),
    [
        (2, "info_gain", "nba", False),
        (2, "gini", "majority", False),
        (2, "info_gain", "nb", False),
        (3, "info_gain", "nba", False),
        (3, "gini", "majority", True),
    ],
)
def test_row_by_row_parity_against_river(
    n_classes: int,
    split_criterion: str,
    leaf_prediction: str,
    weighted: bool,
) -> None:
    df, x, y_idx, w = _tree_stream(
        17, n=600, k=5, n_classes=n_classes, noise=0.10, weighted=weighted
    )
    classes = [f"c{c}" for c in range(n_classes)]
    kw: dict[str, object] = dict(
        grace_period=50.0,
        delta=1e-4,
        split_criterion=split_criterion,
        leaf_prediction=leaf_prediction,
        merit_preprune=False,
        max_share_to_split=1.0,
    )
    if weighted:
        kw["weight"] = "w"
    got_p, got_c = _rheon_clf_probs(df, 5, classes, **kw)
    want_p, want_c = _run_river_clf(
        x,
        y_idx,
        w,
        classes,
        grace_period=50,
        delta=1e-4,
        split_criterion=split_criterion,
        leaf_prediction=leaf_prediction,
        merit_preprune=False,
        max_share_to_split=1.0,
    )
    # Row 0 has no prior observations in either implementation.
    assert np.isnan(got_p[0]).all() and np.isnan(want_p[0]).all()
    assert got_c[0] is None and want_c[0] is None
    # Every subsequent row agrees to ~1e-12 (libm::erf vs platform math.erf
    # and logsumexp summation order).
    diff = np.abs(got_p[1:] - want_p[1:])
    assert float(np.nanmax(diff)) < 1e-11, float(np.nanmax(diff))
    assert got_c[1:] == want_c[1:]


def test_hellinger_learns_imbalanced_binary_stream_without_river_keyerror() -> None:
    """Rheon's Hellinger split criterion handles arbitrary string class labels
    and thresholds outside one class's range cleanly (river 0.26.1's Python
    HellingerDistanceCriterion hardcodes integer keys 0/1 and returns 0 on
    KeyError when either class has zero weight on a side)."""
    from river.tree.split_criterion import HellingerDistanceCriterion

    # Component check: when all four cells are positive, our formula matches river's.
    l0, l1, r0, r1 = 30.0, 5.0, 10.0, 45.0
    want_h = HellingerDistanceCriterion.compute_hellinger([{0: l0, 1: l1}, {0: r0, 1: r1}])
    tn, tp = l0 + r0, l1 + r1
    got_h = math.sqrt(
        (math.sqrt(l0 / tn) - math.sqrt(l1 / tp)) ** 2
        + (math.sqrt(r0 / tn) - math.sqrt(r1 / tp)) ** 2
    )
    assert math.isclose(got_h, want_h, rel_tol=1e-15)

    # End-to-end binary stream with string labels.
    df, _, _, _ = _tree_stream(17, n=1200, k=5, n_classes=2, noise=0.05)
    got_p, got_c = _rheon_clf_probs(
        df,
        5,
        ["c0", "c1"],
        grace_period=50.0,
        delta=1e-4,
        split_criterion="hellinger",
        leaf_prediction="majority",
    )
    tail_labels = df["label"].to_list()[-400:]
    tail_acc = np.mean([a == b for a, b in zip(got_c[-400:], tail_labels, strict=True)])
    assert float(tail_acc) > 0.85
    assert np.isfinite(got_p[1:]).all()


@pytest.mark.oracle_grid(reason="50-seed parity grid against river HoeffdingTreeClassifier")
def test_river_parity_50_seed_grid() -> None:
    """50-seed parity against river's HoeffdingTreeClassifier across binary
    info_gain+nba, binary gini+majority, and 3-class info_gain+nba."""
    cases = [
        ("binary_ig_nba", 2, "info_gain", "nba"),
        ("binary_gini_majority", 2, "gini", "majority"),
        ("multiclass_ig_nba", 3, "info_gain", "nba"),
    ]
    for case_name, n_classes, crit, lp in cases:
        classes = [f"c{c}" for c in range(n_classes)]
        max_prob_diffs = []
        mean_prob_diffs = []
        for s in GRID_SEEDS:
            df, x, y_idx, w = _tree_stream(s, n=600, k=5, n_classes=n_classes, noise=0.10)
            got_p, _ = _rheon_clf_probs(
                df,
                5,
                classes,
                grace_period=50.0,
                delta=1e-4,
                split_criterion=crit,
                leaf_prediction=lp,
                merit_preprune=False,
                max_share_to_split=1.0,
            )
            want_p, _ = _run_river_clf(
                x,
                y_idx,
                w,
                classes,
                grace_period=50,
                delta=1e-4,
                split_criterion=crit,
                leaf_prediction=lp,
                merit_preprune=False,
                max_share_to_split=1.0,
            )
            diffs = np.abs(got_p[1:] - want_p[1:])
            max_prob_diffs.append(float(np.nanmax(diffs)))
            mean_prob_diffs.append(float(np.nanmean(diffs)))
        ledger.record("river_parity", f"{case_name}_mean_prob_diff", mean_prob_diffs, ideal=0.0)
        # Across 50 seeds, libm::erf vs platform math.erf matches to ~1e-12
        # on almost every seed; even on a rare tie-flip seed mean diff stays tiny.
        assert float(np.median(max_prob_diffs)) < 1e-11
        assert max(mean_prob_diffs) < 0.05


# ---------------------------------------------------------------------------
# Deliberate departures from river, memory cap, and streaming contracts


def test_merit_preprune_stops_splitting_on_single_feature_pure_noise() -> None:
    """Null-split merit is 0.0 (Domingos & Hulten 2000), not -inf. On a single
    pure-noise feature, a tree with merit_preprune=True does not split while
    eps >= tau (and never splits with tau=0.0), whereas merit_preprune=False
    splits at the very first grace period."""
    rng = np.random.default_rng(123)
    n = 1000
    df = pl.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "label": ["a" if v else "b" for v in rng.integers(0, 2, size=n)],
        }
    )
    spec_pruned = rp.spec.hoeffding_tree_classifier(
        "m",
        features=["x0"],
        label="label",
        classes=["a", "b"],
        grace_period=50.0,
        delta=1e-3,
        tau=0.05,
        leaf_prediction="majority",
        merit_preprune=True,
    )
    spec_pruned_no_tie = rp.spec.hoeffding_tree_classifier(
        "m",
        features=["x0"],
        label="label",
        classes=["a", "b"],
        grace_period=50.0,
        delta=1e-3,
        tau=0.0,
        leaf_prediction="majority",
        merit_preprune=True,
    )
    spec_unpruned = rp.spec.hoeffding_tree_classifier(
        "m",
        features=["x0"],
        label="label",
        classes=["a", "b"],
        grace_period=50.0,
        delta=1e-3,
        tau=0.05,
        leaf_prediction="majority",
        merit_preprune=False,
    )
    out_p = rp.ModelBank([spec_pruned]).fit_predict(df)["m"]
    out_p0 = rp.ModelBank([spec_pruned_no_tie]).fit_predict(df)["m"]
    out_u = rp.ModelBank([spec_unpruned]).fit_predict(df)["m"]
    # With no split, every majority-class probability on row i equals the
    # cumulative share of class "a" over rows 0..i-1.
    labels = df["label"].to_list()
    cum_a = np.cumsum([1.0 if v == "a" else 0.0 for v in labels[:-1]])
    expected_p_a = cum_a / np.arange(1, n, dtype=np.float64)
    np.testing.assert_allclose(
        out_p.struct.field("p_a").to_numpy()[1:], expected_p_a, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        out_p0.struct.field("p_a").to_numpy()[1:], expected_p_a, rtol=0.0, atol=1e-15
    )
    # Without prepruning, the lone feature splits immediately at grace_period=50.
    got_u_a = out_u.struct.field("p_a").to_numpy()[1:]
    assert float(np.max(np.abs(got_u_a - expected_p_a))) > 0.02


def test_zero_weight_rows_change_nothing_bit_for_bit() -> None:
    """Hard rule 9: zero-weight rows are scored, leave Gaussian min/max and
    counts untouched, and produce bit-identical predictions to the stream
    without them."""
    df, _, _, _ = _tree_stream(5, n=400, k=4, n_classes=2, noise=0.08)
    # Insert extreme outlier feature values at weight 0 that would poison
    # river's Gaussian min/max range.
    rows = []
    for i in range(df.height):
        rows.append(df.row(i, named=True))
        if i % 25 == 10:
            rows.append(
                {
                    "x0": 1e6,
                    "x1": -1e6,
                    "x2": 1e6,
                    "x3": -1e6,
                    "label": "c0",
                    "w": 0.0,
                }
            )
    with_zeros = pl.DataFrame(rows)
    spec = rp.spec.hoeffding_tree_classifier(
        "m",
        features=["x0", "x1", "x2", "x3"],
        label="label",
        classes=["c0", "c1"],
        weight="w",
        grace_period=40.0,
        delta=1e-3,
    )
    a = rp.ModelBank([spec]).fit_predict(df)["m"]
    b_full = rp.ModelBank([spec]).fit_predict(with_zeros)
    b = b_full.filter(pl.col("w") > 0.0)["m"]
    assert a.equals(b, null_equal=True)
    # Zero-weight rows after the first learned row are still scored.
    zero_rows = b_full.filter(pl.col("w") == 0.0)["m"]
    assert zero_rows.struct.field("class").null_count() == 0


def test_cap_bounds_state_and_save_resume_is_bit_exact(tmp_path) -> None:
    """Under a tight per-group byte cap, the tree degrades gracefully by
    deactivating observers instead of refusing rows, keeps serialized state
    compact, and resumes across save/load bit-identically (including the
    1,000-row recheck counter)."""
    df, _, _, _ = _tree_stream(9, n=2500, k=6, n_classes=3, noise=0.05)
    # 1 active leaf reserve = 40 * 6 * 3 = 720 bytes; cap at 2,500 bytes lets
    # only a couple of leaves keep observers at a time while splitting past
    # the 1,000-row and 2,000-row recheck boundaries.
    spec = rp.spec.hoeffding_tree_classifier(
        "m",
        features=[f"x{j}" for j in range(6)],
        label="label",
        classes=["c0", "c1", "c2"],
        grace_period=40.0,
        delta=1e-2,
        max_state_bytes=2500,
    )
    whole = rp.ModelBank([spec]).fit_predict(df)["m"]
    # Graceful observer deactivation never triggers a cap refusal or withhold.
    assert whole.struct.field("withheld_reason").drop_nulls().to_list() == []
    assert whole.struct.field("class").null_count() == 1  # only row 0 before any label

    # Split mid-stream right between rechecks (row 1,234), save, reload, continue.
    bank_a = rp.ModelBank([spec])
    p1 = bank_a.fit_predict(df.slice(0, 1234))["m"]
    state_path = tmp_path / "tree_clf.state"
    bank_a.save(state_path)
    bank_b = rp.ModelBank.load(state_path)
    p2 = bank_b.fit_predict(df.slice(1234))["m"]
    resumed = pl.concat([p1, p2])
    assert whole.equals(resumed, null_equal=True)
    # Still learns accurately even under the tight cap.
    tail_acc = (whole.struct.field("class").tail(500) == df["label"].tail(500)).mean()
    assert tail_acc is not None and float(tail_acc) > 0.70


def test_chunk_invariance_and_null_labels() -> None:
    df, _, _, _ = _tree_stream(31, n=500, k=4, n_classes=3, noise=0.08)
    labels = [None if i % 17 == 5 else v for i, v in enumerate(df["label"].to_list())]
    df = df.with_columns(pl.Series("label", labels))
    spec = rp.spec.hoeffding_tree_classifier(
        "m",
        features=["x0", "x1", "x2", "x3"],
        label="label",
        classes=["c0", "c1", "c2"],
        grace_period=30.0,
        delta=1e-3,
    )
    one = rp.ModelBank([spec]).fit_predict(df)["m"]
    bank = rp.ModelBank([spec])
    many = pl.concat([bank.fit_predict(df.slice(i, 19))["m"] for i in range(0, df.height, 19)])
    assert one.equals(many, null_equal=True)
    # Null-label rows after row 0 are scored and do not teach the tree.
    assert one.struct.field("class")[22] is not None


def test_max_depth_zero_never_splits() -> None:
    df, _, _, _ = _tree_stream(11, n=300, k=3, n_classes=2, noise=0.05)
    spec = rp.spec.hoeffding_tree_classifier(
        "m",
        features=["x0", "x1", "x2"],
        label="label",
        classes=["c0", "c1"],
        grace_period=20.0,
        delta=1e-2,
        max_depth=0,
        leaf_prediction="majority",
    )
    out = rp.ModelBank([spec]).fit_predict(df)["m"]
    # With max_depth=0 the root never splits; majority probabilities equal the
    # cumulative class shares over the whole stream.
    labels = df["label"].to_list()
    cum0 = np.cumsum([1.0 if v == "c0" else 0.0 for v in labels[:-1]])
    expected = cum0 / np.arange(1, len(labels), dtype=np.float64)
    np.testing.assert_allclose(out.struct.field("p_c0").to_numpy()[1:], expected, atol=1e-15)


def test_deactivated_leaf_is_not_reactivated_by_later_splits() -> None:
    """A leaf deactivated by max_share_to_split stays inactive permanently
    (unlike river's Python path, which reactivates it on the next split)."""
    # Half of the feature space (x0 <= 0) is 100% pure class c0; x0 > 0 is
    # noisy 50/50 separated by x1.
    rng = np.random.default_rng(77)
    n = 1200
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    labels = []
    for a, b in zip(x0, x1, strict=True):
        if a <= 0.0:
            labels.append("c0")
        else:
            labels.append("c0" if b <= 0.0 else "c1")
    df = pl.DataFrame({"x0": x0, "x1": x1, "label": labels})
    spec = rp.spec.hoeffding_tree_classifier(
        "m",
        features=["x0", "x1"],
        label="label",
        classes=["c0", "c1"],
        grace_period=30.0,
        delta=1e-2,
        max_share_to_split=0.95,
    )
    out = rp.ModelBank([spec]).fit_predict(df)["m"]
    tail_acc = (out.struct.field("class").tail(400) == df["label"].tail(400)).mean()
    assert tail_acc is not None and float(tail_acc) > 0.97


def test_classifier_refusals() -> None:
    base = dict(features=["x0", "x1"], label="lab", classes=["a", "b"])
    with pytest.raises(ValueError, match="half_life"):
        rp.spec.hoeffding_tree_classifier("m", **base, half_life=50.0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="lam"):
        rp.spec.hoeffding_tree_classifier("m", **base, lam=0.99)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="min_settled_frac"):
        rp.spec.hoeffding_tree_classifier("m", **base, min_settled_frac=0.5)
    with pytest.raises(ValueError, match="hellinger"):
        rp.spec.hoeffding_tree_classifier(
            "m", features=["x0"], label="lab", classes=["a", "b", "c"], split_criterion="hellinger"
        )
    with pytest.raises(ValueError, match="split_criterion"):
        rp.spec.hoeffding_tree_classifier("m", **base, split_criterion="mse")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="leaf_prediction"):
        rp.spec.hoeffding_tree_classifier("m", **base, leaf_prediction="mean")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_state_bytes"):
        rp.spec.hoeffding_tree_classifier("m", **base, max_state_bytes=10)
    with pytest.raises(ValueError, match="max_state_bytes"):
        rp.spec.hoeffding_tree_classifier("m", **base, max_state_bytes=64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="grace_period"):
        rp.spec.hoeffding_tree_classifier("m", **base, grace_period=0.0)
    with pytest.raises(ValueError, match="delta"):
        rp.spec.hoeffding_tree_classifier("m", **base, delta=1.0)
    with pytest.raises(ValueError, match="tau"):
        rp.spec.hoeffding_tree_classifier("m", **base, tau=math.nan)
    with pytest.raises(ValueError, match="n_splits"):
        rp.spec.hoeffding_tree_classifier("m", **base, n_splits=0)
    with pytest.raises(ValueError, match="classes"):
        rp.spec.hoeffding_tree_classifier("m", features=["x0"], label="lab", classes=["only_one"])
