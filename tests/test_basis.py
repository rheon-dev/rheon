"""Stateless basis-expansion operators in ``rheon.polars.ops`` (PLAN 572).

Covers ``bspline``, ``natural_spline``, ``hinge``, and ``interactions``:

- exact agreement with ``scipy.interpolate.BSpline.design_matrix``,
  ``scipy.linalg.qr`` natural-spline projection, ``scipy.interpolate.CubicSpline``,
  ``sklearn.preprocessing.SplineTransformer``, and
  ``sklearn.preprocessing.PolynomialFeatures``;
- boundary properties (partition of unity, non-negativity, zero 2nd derivative
  at boundary knots, exact linear/constant/null extrapolation);
- missing value propagation (``null -> null``, ``NaN -> NaN``), streaming and
  chunk invariance, ``ModelBank`` end-to-end integration (nonlinear ridge and
  logistic spline calibration), zero-NumPy runtime execution, input validation,
  and a 50-seed oracle-ledger grid.
"""

from __future__ import annotations

import ast
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import polars.testing
import pytest
from scipy.interpolate import BSpline, CubicSpline
from scipy.linalg import qr
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

import ledger
import rheon
from rheon import polars as rp
from rheon.polars._basis import _bspline_scalar_eval, _nullspace_qr_2rows
from rheon.polars.ops import bspline, hinge, interactions, natural_spline

TIER = "mixed"

REPO = Path(__file__).resolve().parent.parent
GRID_SEEDS = list(range(50))


# ---------------------------------------------------------------------------
# Third-party oracle helpers (scipy + scikit-learn)
# ---------------------------------------------------------------------------


def _scipy_clamped_bspline(
    x: np.ndarray,
    knots: np.ndarray,
    degree: int,
    *,
    include_bias: bool = True,
    extrapolate: str = "constant",
) -> np.ndarray:
    """Clamped B-spline design matrix from ``scipy.interpolate.BSpline``."""
    k_lo, k_hi = float(knots[0]), float(knots[-1])
    t = np.concatenate([[k_lo] * degree, knots, [k_hi] * degree])
    m = len(t) - degree - 1
    x_cl = np.clip(x, k_lo, k_hi)
    mat = BSpline.design_matrix(x_cl, t, degree).toarray()
    if extrapolate == "linear" and degree >= 1:
        lo_mask = x < k_lo
        hi_mask = x > k_hi
        if lo_mask.any() or hi_mask.any():
            for i in range(m):
                c = np.zeros(m)
                c[i] = 1.0
                sp = BSpline(t, c, degree)
                sp1 = sp.derivative(1)
                if lo_mask.any():
                    mat[lo_mask, i] = sp(k_lo) + (x[lo_mask] - k_lo) * sp1(k_lo)
                if hi_mask.any():
                    mat[hi_mask, i] = sp(k_hi) + (x[hi_mask] - k_hi) * sp1(k_hi)
    elif extrapolate == "null":
        out_mask = (x < k_lo) | (x > k_hi)
        mat[out_mask, :] = np.nan
    return mat if include_bias else mat[:, 1:]


def _scipy_natural_spline_qr(
    x: np.ndarray,
    knots: np.ndarray,
    *,
    include_bias: bool = False,
) -> np.ndarray:
    """Natural cubic B-spline via ``scipy.interpolate.BSpline`` + ``scipy.linalg.qr``."""
    k_lo, k_hi = float(knots[0]), float(knots[-1])
    t = np.concatenate([[k_lo] * 3, knots, [k_hi] * 3])
    m = len(t) - 4
    c_mat = np.zeros((2, m))
    b_lo = np.zeros(m)
    b1_lo = np.zeros(m)
    b_hi = np.zeros(m)
    b1_hi = np.zeros(m)
    for i in range(m):
        coef = np.zeros(m)
        coef[i] = 1.0
        sp = BSpline(t, coef, 3)
        c_mat[0, i] = sp.derivative(2)(k_lo)
        c_mat[1, i] = sp.derivative(2)(k_hi)
        b_lo[i] = sp(k_lo)
        b1_lo[i] = sp.derivative(1)(k_lo)
        b_hi[i] = sp(k_hi)
        b1_hi[i] = sp.derivative(1)(k_hi)
    start = 0 if include_bias else 1
    c_sub = c_mat[:, start:]
    c_norm = c_sub / np.linalg.norm(c_sub, axis=1, keepdims=True)
    q_mat, _ = qr(c_norm.T, mode="full")
    z_mat = q_mat[:, 2:]

    x_cl = np.clip(x, k_lo, k_hi)
    b_in = BSpline.design_matrix(x_cl, t, 3).toarray()[:, start:]
    out = b_in @ z_mat
    lo_mask = x < k_lo
    hi_mask = x > k_hi
    if lo_mask.any():
        out[lo_mask] = (b_lo[start:] @ z_mat) + np.outer(x[lo_mask] - k_lo, b1_lo[start:] @ z_mat)
    if hi_mask.any():
        out[hi_mask] = (b_hi[start:] @ z_mat) + np.outer(x[hi_mask] - k_hi, b1_hi[start:] @ z_mat)
    return out


def _align_column_signs(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Align ±1 Householder reflector sign per column for direct element-wise comparison."""
    aligned = ref.copy()
    for j in range(aligned.shape[1]):
        if float(np.dot(got[:, j], aligned[:, j])) < 0.0:
            aligned[:, j] = -aligned[:, j]
    return aligned


# ---------------------------------------------------------------------------
# Unit & mathematical property tests: bspline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("include_bias", [True, False])
def test_bspline_matches_scipy_design_matrix(degree: int, include_bias: bool) -> None:
    knots = np.array([-3.0, -1.2, 0.0, 0.4, 2.5, 6.0])
    rng = np.random.default_rng(100 + degree)
    xs = np.concatenate([knots, rng.uniform(knots[0], knots[-1], 150)])
    df = pl.DataFrame({"x": xs})
    got = df.select(
        bspline("x", knots.tolist(), degree=degree, include_bias=include_bias)
    ).to_numpy()
    want = _scipy_clamped_bspline(xs, knots, degree, include_bias=include_bias)
    assert got.shape == (len(xs), len(knots) + degree - 1 - (0 if include_bias else 1))
    assert float(np.max(np.abs(got - want))) < 1e-13
    # Non-negativity and partition of unity inside [k_min, k_max].
    assert float(np.min(got)) >= -1e-15
    if include_bias:
        assert float(np.max(np.abs(got.sum(axis=1) - 1.0))) < 1e-13


@pytest.mark.parametrize("extrapolate", ["constant", "linear", "null"])
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_bspline_extrapolation_modes(extrapolate: str, degree: int) -> None:
    knots = np.array([0.0, 0.3, 0.75, 1.0])
    xs = np.array([-1.5, -0.25, 0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.4, 3.0])
    df = pl.DataFrame({"x": xs})
    out = df.select(
        bspline("x", knots.tolist(), degree=degree, include_bias=True, extrapolate=extrapolate)
    )
    if extrapolate == "null":
        for row_idx in (0, 1, 8, 9):
            assert all(v is None for v in out.row(row_idx))
        inner = out.slice(2, 6).to_numpy()
        want_inner = _scipy_clamped_bspline(xs[2:8], knots, degree, include_bias=True)
        assert float(np.max(np.abs(inner - want_inner))) < 1e-13
    else:
        got = out.to_numpy()
        want = _scipy_clamped_bspline(xs, knots, degree, include_bias=True, extrapolate=extrapolate)
        assert float(np.max(np.abs(got - want))) < 1e-13
        # Partition of unity holds even under constant and linear extrapolation!
        assert float(np.max(np.abs(got.sum(axis=1) - 1.0))) < 1e-13


def test_bspline_two_knots_boundary_only() -> None:
    knots = [2.0, 5.0]
    xs = np.linspace(1.0, 6.0, 41)
    df = pl.DataFrame({"x": xs})
    # Degree 0 with include_bias=True -> constant 1.0 everywhere.
    got0 = df.select(bspline("x", knots, degree=0, include_bias=True)).to_numpy()
    np.testing.assert_allclose(got0, np.ones((len(xs), 1)), atol=1e-15)
    # Degree 3 matches scipy Bernstein basis on [2, 5].
    for mode in ("constant", "linear"):
        got3 = df.select(
            bspline("x", knots, degree=3, include_bias=False, extrapolate=mode)
        ).to_numpy()
        want3 = _scipy_clamped_bspline(xs, np.array(knots), 3, include_bias=False, extrapolate=mode)
        assert float(np.max(np.abs(got3 - want3))) < 1e-13


def test_bspline_spans_same_subspace_as_sklearn_spline_transformer() -> None:
    rng = np.random.default_rng(7)
    knots = np.array([-2.0, -0.5, 0.2, 1.1, 3.0])
    xs = rng.uniform(knots[0], knots[-1], 250)
    df = pl.DataFrame({"x": xs})
    for degree in (1, 2, 3):
        got = df.select(bspline("x", knots.tolist(), degree=degree, include_bias=True)).to_numpy()
        st = SplineTransformer(
            knots=knots[:, None], degree=degree, include_bias=True, extrapolation="constant"
        ).fit(xs[:, None])
        sk = st.transform(xs[:, None])
        assert got.shape == sk.shape
        # Exact mutual subspace containment: project sk onto got and got onto sk.
        c1, *_ = np.linalg.lstsq(got, sk, rcond=None)
        c2, *_ = np.linalg.lstsq(sk, got, rcond=None)
        assert float(np.max(np.abs(got @ c1 - sk))) < 1e-13
        assert float(np.max(np.abs(sk @ c2 - got))) < 1e-13


# ---------------------------------------------------------------------------
# Unit & mathematical property tests: natural_spline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "knots",
    [
        [0.0, 1.0],  # boundary knots only: pure linear subspace
        [-1.0, 0.2, 1.5],  # 1 interior knot
        [-4.0, -1.5, 0.0, 0.8, 2.2, 5.0],  # non-uniform interior knots
        [0.0, 1e-6, 3e-6, 1e-5],  # tiny scale
        [0.0, 1e6, 3e6, 1e7],  # huge scale
    ],
)
@pytest.mark.parametrize("include_bias", [False, True])
def test_natural_spline_matches_scipy_qr_and_properties(
    knots: list[float], include_bias: bool
) -> None:
    k_arr = np.array(knots, dtype=float)
    span = float(k_arr[-1] - k_arr[0])
    rng = np.random.default_rng(2026)
    xs = np.concatenate(
        [
            np.linspace(k_arr[0] - 0.8 * span, k_arr[0] - 0.05 * span, 20),
            k_arr,
            rng.uniform(k_arr[0], k_arr[-1], 120),
            np.linspace(k_arr[-1] + 0.05 * span, k_arr[-1] + 0.8 * span, 20),
        ]
    )
    df = pl.DataFrame({"x": xs})
    got = df.select(natural_spline("x", knots, include_bias=include_bias)).to_numpy()
    want_raw = _scipy_natural_spline_qr(xs, k_arr, include_bias=include_bias)
    want = _align_column_signs(got, want_raw)

    expected_cols = len(knots) if include_bias else len(knots) - 1
    assert got.shape == (len(xs), expected_cols)
    assert float(np.max(np.abs(got - want))) < 1e-12

    # When include_bias=False, every basis function vanishes at k_0 -> no intercept collinearity.
    if not include_bias:
        at_lo = (
            pl.DataFrame({"x": [k_arr[0]]})
            .select(natural_spline("x", knots, include_bias=False))
            .to_numpy()
        )
        assert float(np.max(np.abs(at_lo))) < 1e-14

    # Exact linear extrapolation outside [k_min, k_max]: 2nd finite difference is zero.
    h = 0.25 * span
    for pts in (
        [k_arr[0] - 2 * h, k_arr[0] - h, k_arr[0]],
        [k_arr[-1], k_arr[-1] + h, k_arr[-1] + 2 * h],
    ):
        vals = (
            pl.DataFrame({"x": pts})
            .select(natural_spline("x", knots, include_bias=include_bias))
            .to_numpy()
        )
        d2_lin = vals[0] - 2.0 * vals[1] + vals[2]
        assert float(np.max(np.abs(d2_lin))) < 1e-12 * max(1.0, float(np.max(np.abs(vals))))

    # Check exact QR nullspace orthogonality and zero 2nd derivatives at boundary knots.
    _, _, d2_lo = _bspline_scalar_eval(k_arr[0], tuple(knots), 3)
    _, _, d2_hi = _bspline_scalar_eval(k_arr[-1], tuple(knots), 3)
    start = 0 if include_bias else 1
    c_mat = np.array([d2_lo[start:], d2_hi[start:]])
    z_mat = np.array(_nullspace_qr_2rows(d2_lo[start:], d2_hi[start:]))
    np.testing.assert_allclose(z_mat.T @ z_mat, np.eye(z_mat.shape[1]), atol=1e-14)
    c_norm = c_mat / np.linalg.norm(c_mat, axis=1, keepdims=True)
    assert float(np.max(np.abs(c_norm @ z_mat))) < 1e-14


def test_natural_spline_matches_scipy_cubicspline_natural_subspace() -> None:
    rng = np.random.default_rng(572)
    knots = np.array([-2.0, -0.8, 0.1, 0.9, 2.3, 4.0])
    y_knots = rng.standard_normal(len(knots))
    cs = CubicSpline(knots, y_knots, bc_type="natural")
    xs = np.linspace(knots[0], knots[-1], 300)
    y_true = cs(xs)
    df = pl.DataFrame({"x": xs})

    # With include_bias=True, N(x) alone spans every natural cubic spline on these knots.
    n_full = df.select(natural_spline("x", knots.tolist(), include_bias=True)).to_numpy()
    coef_full, *_ = np.linalg.lstsq(n_full, y_true, rcond=None)
    assert float(np.max(np.abs(n_full @ coef_full - y_true))) < 1e-13

    # With include_bias=False + intercept column 1, exact same span!
    n_nb = df.select(natural_spline("x", knots.tolist(), include_bias=False)).to_numpy()
    design = np.column_stack([np.ones(len(xs)), n_nb])
    coef_nb, *_ = np.linalg.lstsq(design, y_true, rcond=None)
    assert float(np.max(np.abs(design @ coef_nb - y_true))) < 1e-13


# ---------------------------------------------------------------------------
# Unit tests: hinge & interactions
# ---------------------------------------------------------------------------


def test_hinge_emits_only_positive_parts_and_no_raw_x() -> None:
    knots = [-1.5, 0.0, 2.25]
    xs = np.array([-3.0, -1.5, -0.5, 0.0, 1.0, 2.25, 4.5])
    df = pl.DataFrame({"x": xs})
    out = df.select(hinge("x", knots))
    assert out.columns == ["x_hinge_0", "x_hinge_1", "x_hinge_2"]
    assert "x" not in out.columns
    want = np.column_stack([np.maximum(0.0, xs - k) for k in knots])
    np.testing.assert_array_equal(out.to_numpy(), want)

    # Single knot and custom prefix.
    one = df.select(hinge("x", [0.0], prefix="pos"))
    assert one.columns == ["pos_0"]
    np.testing.assert_array_equal(one["pos_0"].to_numpy(), np.maximum(0.0, xs))


@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("include_self", [False, True])
def test_interactions_matches_sklearn_polynomial_features(degree: int, include_self: bool) -> None:
    rng = np.random.default_rng(314 + degree)
    names = ["a", "b", "c", "d"]
    mat = rng.standard_normal((80, len(names)))
    df = pl.DataFrame({name: mat[:, i] for i, name in enumerate(names)})

    exprs = interactions(names, degree=degree, include_self=include_self)
    out = df.select(exprs)
    pf = PolynomialFeatures(degree=degree, interaction_only=not include_self, include_bias=False)
    want = pf.fit_transform(mat)[:, len(names) :]
    assert out.width == want.shape[1]
    assert float(np.max(np.abs(out.to_numpy() - want))) < 1e-13


def test_interactions_naming_sep_and_single_col_self() -> None:
    df = pl.DataFrame({"a": [2.0, -3.0], "b": [5.0, 4.0]})
    out = df.select(interactions(["a", "b"], sep="*"))
    assert out.columns == ["a*b"]
    assert out["a*b"].to_list() == [10.0, -12.0]

    sq = df.select(interactions(["a"], degree=3, include_self=True))
    assert sq.columns == ["a_x_a", "a_x_a_x_a"]
    assert sq["a_x_a"].to_list() == [4.0, 9.0]
    assert sq["a_x_a_x_a"].to_list() == [8.0, -27.0]


# ---------------------------------------------------------------------------
# Missing values (null, NaN, inf), expressions, dtypes, streaming & chunking
# ---------------------------------------------------------------------------


def test_null_nan_and_inf_handling_across_all_operators() -> None:
    df = pl.DataFrame(
        {
            "x": [None, float("nan"), float("-inf"), -1.0, 0.5, 2.0, float("inf")],
            "y": [1.0, 2.0, 3.0, None, float("nan"), 4.0, 5.0],
        }
    )

    for mode in ("constant", "linear", "null"):
        out = df.select(bspline("x", [0.0, 0.5, 1.0], extrapolate=mode))
        for col_name in out.columns:
            vals = out[col_name].to_list()
            assert vals[0] is None, f"bspline({mode}) turned null into {vals[0]}"
            assert isinstance(vals[1], float) and math.isnan(vals[1]), (
                f"bspline({mode}) turned NaN into {vals[1]}"
            )
            assert math.isfinite(vals[4]), f"bspline({mode}) interior value non-finite"
            if mode == "null":
                assert vals[2] is None and vals[3] is None and vals[5] is None and vals[6] is None
            elif mode == "constant":
                assert math.isfinite(vals[2]) and math.isfinite(vals[6])
            else:
                # Linear extrapolation never emits NaN from 0 * inf.
                assert not math.isnan(vals[2]) and not math.isnan(vals[6])

    ns_out = df.select(natural_spline("x", [0.0, 0.5, 1.0]))
    for col_name in ns_out.columns:
        vals = ns_out[col_name].to_list()
        assert vals[0] is None
        assert isinstance(vals[1], float) and math.isnan(vals[1])
        assert not math.isnan(vals[2]) and not math.isnan(vals[6])
        assert math.isfinite(vals[3]) and math.isfinite(vals[4]) and math.isfinite(vals[5])

    h_out = df.select(hinge("x", [0.0, 1.0]))
    assert h_out["x_hinge_0"][0] is None
    assert math.isnan(h_out["x_hinge_0"][1])
    assert h_out["x_hinge_0"][2] == 0.0
    assert math.isinf(h_out["x_hinge_0"][6]) and h_out["x_hinge_0"][6] > 0

    i_out = df.select(interactions(["x", "y"]))
    vals = i_out["x_x_y"].to_list()
    assert vals[0] is None and vals[3] is None
    assert math.isnan(vals[1]) and math.isnan(vals[4])
    assert vals[5] == 8.0


def test_expr_inputs_prefix_and_integer_float32_dtypes() -> None:
    df = pl.DataFrame(
        {
            "xi": pl.Series([0, 1, 2, 3, 4], dtype=pl.Int64),
            "xf": pl.Series([0.0, 0.25, 0.5, 0.75, 1.0], dtype=pl.Float32),
        }
    )
    out = df.select(
        [
            *bspline(pl.col("xi") * 0.25, [0.0, 0.5, 1.0], prefix="scaled"),
            *natural_spline((pl.col("xf") + 0.0).alias("z"), [0.0, 0.5, 1.0]),
            *hinge(pl.col("xi"), [1.0, 3.0], prefix="h"),
            *interactions([pl.col("xi").alias("a"), (pl.col("xf") * 2.0).alias("b")]),
        ]
    )
    assert "scaled_0" in out.columns and "z_ns_0" in out.columns
    assert "h_0" in out.columns and "a_x_b" in out.columns
    assert all(dt == pl.Float64 for dt in out.dtypes)
    # xi * 0.25 equals xf on every row -> identical B-spline values.
    direct = df.select(bspline("xf", [0.0, 0.5, 1.0], prefix="scaled"))
    np.testing.assert_allclose(out.select(direct.columns).to_numpy(), direct.to_numpy(), atol=1e-15)


def test_basis_operators_stream_and_are_chunk_invariant() -> None:
    rng = np.random.default_rng(99)
    n = 5_000
    df = pl.DataFrame(
        {
            "x": rng.uniform(-1.5, 2.5, n),
            "z": rng.standard_normal(n),
            "w": rng.standard_normal(n),
        }
    )
    exprs = [
        *bspline("x", [0.0, 0.5, 1.0, 1.5], extrapolate="linear"),
        *natural_spline("x", [0.0, 0.5, 1.0, 1.5]),
        *hinge("x", [0.0, 1.0]),
        *interactions(["x", "z", "w"], degree=2),
    ]
    whole = df.select(exprs)
    streamed = df.lazy().select(exprs).collect(engine="streaming")
    chunked = pl.concat([df.slice(i, 317) for i in range(0, n, 317)], rechunk=False).select(exprs)
    assert whole.equals(streamed)
    assert whole.equals(chunked)


# ---------------------------------------------------------------------------
# End-to-end ModelBank integration
# ---------------------------------------------------------------------------


def test_model_bank_integration_nonlinear_ridge_and_spline_calibrator() -> None:
    rng = np.random.default_rng(1234)
    n = 2_000
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.standard_normal(n)
    # Smooth nonlinear regression target + hinge + interaction.
    y_reg = (
        np.sin(2.0 * x)
        + 0.5 * np.maximum(0.0, x - 0.2)
        + 0.4 * x * z
        + 0.05 * rng.standard_normal(n)
    )
    # Binary calibration target with nonlinear log-odds.
    logits = -0.3 + 2.0 * x - 1.2 * x**2
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_bin = (rng.random(n) < probs).astype(float)

    df = pl.DataFrame({"x": x, "z": z, "y_reg": y_reg, "y_bin": y_bin})

    knots = [-1.0, -0.5, 0.0, 0.5, 1.0]
    basis_exprs = [
        *natural_spline("x", knots, prefix="ns_x"),
        *hinge("x", [0.2], prefix="h_x"),
        *interactions(["x", "z"]),
    ]
    feat_names = [e.meta.output_name() for e in basis_exprs]
    ns_names = [f"ns_x_{i}" for i in range(len(knots) - 1)]

    spec_reg = rp.spec.ewridge(
        "nl_ridge",
        targets=["y_reg"],
        features=feat_names,
        half_life=float("inf"),
        ridge=1e-4,
        min_weight=20.0,
        coef_every=0.0,
    )
    spec_cal = rp.spec.sgd(
        "calibrator",
        targets=["y_bin"],
        features=ns_names,
        loss="logistic",
        half_life=float("inf"),
        learning_rate=0.05,
        min_weight=20.0,
        coef_every=0.0,
    )

    # Full LazyFrame streaming query vs chunked ModelBank batches -> bit-identical!
    lf_out = (
        df.lazy()
        .with_columns(basis_exprs)
        .rheon.fit_predict([spec_reg, spec_cal], chunk_size=137)
        .collect()
    )
    expanded = df.with_columns(basis_exprs)
    eager_out = rp.ModelBank([spec_reg, spec_cal]).fit_predict(expanded)
    polars.testing.assert_frame_equal(lf_out, eager_out, check_exact=True)

    # Verify the nonlinear ridge fit beats a linear baseline decisively on the tail.
    tail = lf_out.tail(500)
    pred_reg = tail["nl_ridge"].struct.field("pred_y_reg").to_numpy()
    y_tail = tail["y_reg"].to_numpy()
    r2 = 1.0 - float(np.var(y_tail - pred_reg) / np.var(y_tail))
    assert r2 > 0.98, f"expected nonlinear spline+hinge+interaction R2 > 0.98, got {r2:.4f}"

    # Verify logistic spline calibrator produces valid probabilities correlated with true probs.
    pred_p = tail["calibrator"].struct.field("pred_y_bin").to_numpy()
    assert np.all((pred_p > 0.0) & (pred_p < 1.0))
    corr = float(np.corrcoef(pred_p, probs[-500:])[0, 1])
    assert corr > 0.85, f"expected calibrated probability correlation > 0.85, got {corr:.4f}"


# ---------------------------------------------------------------------------
# Zero-NumPy runtime execution check
# ---------------------------------------------------------------------------


def test_basis_operators_run_without_numpy() -> None:
    """``rheon[numpy]`` is optional; ``_basis.py`` imports no NumPy and executes with it blocked."""
    basis_path = REPO / "python" / "rheon" / "polars" / "_basis.py"
    tree = ast.parse(basis_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "numpy" for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != "numpy"

    code = textwrap.dedent(
        f"""
        import importlib.util
        import sys

        class WithoutNumpy:
            @staticmethod
            def find_spec(name, path=None, target=None):
                if name.split(".")[0] == "numpy":
                    raise ModuleNotFoundError(f"No module named {{name!r}} (blocked)", name=name)
                return None

        assert "numpy" not in sys.modules
        sys.meta_path.insert(0, WithoutNumpy)

        import polars as pl
        import rheon._errors  # neutral module, no numpy

        spec = importlib.util.spec_from_file_location("_basis_probe", {str(basis_path)!r})
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        df = pl.DataFrame({{
            "x": [-0.5, 0.0, 0.25, 0.5, 1.0, 1.5],
            "z": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }})
        out = df.select([
            *mod.bspline("x", [0.0, 0.5, 1.0], extrapolate="linear"),
            *mod.natural_spline("x", [0.0, 0.5, 1.0]),
            *mod.hinge("x", [0.0, 0.5]),
            *mod.interactions(["x", "z"]),
        ])
        assert out.height == 6 and out.width == 9
        assert "numpy" not in sys.modules
        print("ok")
        """
    )
    res = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
        cwd=REPO,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# Ops-home & validation refusal tests
# ---------------------------------------------------------------------------


def test_basis_operators_live_in_ops_alone() -> None:
    for name, fn in (
        ("bspline", bspline),
        ("natural_spline", natural_spline),
        ("hinge", hinge),
        ("interactions", interactions),
    ):
        assert getattr(rp.ops, name) is fn
        assert name in rp.ops.__all__
        assert not hasattr(rp, name)
        assert not hasattr(rheon, name)


def test_basis_refusals() -> None:
    # Bad col / prefix
    with pytest.raises(TypeError, match="col must be"):
        bspline(123, [0.0, 1.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty column name"):
        bspline("", [0.0, 1.0])
    with pytest.raises(ValueError, match="multiple outputs"):
        bspline(pl.col("a", "b"), [0.0, 1.0])
    with pytest.raises(TypeError, match="prefix must be"):
        hinge("x", [0.0], prefix=42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty string"):
        natural_spline("x", [0.0, 1.0], prefix="")

    # Bad knots
    with pytest.raises(TypeError, match="sequence of numbers"):
        bspline("x", "01")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 2"):
        bspline("x", [0.0])
    with pytest.raises(ValueError, match="at least 2"):
        natural_spline("x", [0.0])
    with pytest.raises(ValueError, match="at least 1"):
        hinge("x", [])
    with pytest.raises(TypeError, match="real number"):
        hinge("x", [True])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite"):
        bspline("x", [0.0, float("inf")])
    with pytest.raises(ValueError, match="finite"):
        bspline("x", [0.0, float("nan")])
    with pytest.raises(ValueError, match="strictly increasing"):
        bspline("x", [0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        natural_spline("x", [1.0, 0.0])

    # Bad degree / include_bias / extrapolate
    with pytest.raises(TypeError, match="degree must be an int"):
        bspline("x", [0.0, 1.0], degree=True)
    with pytest.raises(ValueError, match="between 0 and"):
        bspline("x", [0.0, 1.0], degree=-1)
    with pytest.raises(ValueError, match="between 0 and"):
        bspline("x", [0.0, 1.0], degree=11)
    with pytest.raises(ValueError, match="0 basis columns"):
        bspline("x", [0.0, 1.0], degree=0, include_bias=False)
    with pytest.raises(TypeError, match="include_bias must be"):
        natural_spline("x", [0.0, 1.0], include_bias=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="extrapolate must be a str"):
        bspline("x", [0.0, 1.0], extrapolate=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="extrapolate must be one of"):
        bspline("x", [0.0, 1.0], extrapolate="periodic")  # type: ignore[arg-type]

    # Bad interactions arguments
    with pytest.raises(TypeError, match="cols must be a sequence"):
        interactions("ab")
    with pytest.raises(TypeError, match="cols must be a sequence"):
        interactions(pl.col("a"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="between 2 and"):
        interactions(["a", "b"], degree=1)
    with pytest.raises(ValueError, match="between 2 and"):
        interactions(["a", "b"], degree=7)
    with pytest.raises(TypeError, match="include_self must be"):
        interactions(["a", "b"], include_self="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sep must be a str"):
        interactions(["a", "b"], sep=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty string"):
        interactions(["a", "b"], sep="")
    with pytest.raises(ValueError, match="non-empty column name"):
        interactions(["", "b"])
    with pytest.raises(ValueError, match="multiple outputs"):
        interactions([pl.col("a", "b"), "c"])
    with pytest.raises(ValueError, match="duplicate column"):
        interactions(["a", "a"])
    with pytest.raises(ValueError, match="fewer than degree=3"):
        interactions(["a", "b"], degree=3, include_self=False)
    with pytest.raises(ValueError, match="at least 1 column"):
        interactions([], include_self=True)
    with pytest.raises(ValueError, match="collides"):
        interactions(["a", "b_x_c", "a_x_b", "c"], sep="_x_")


# ---------------------------------------------------------------------------
# 50-seed oracle ledger grid against scipy & scikit-learn
# ---------------------------------------------------------------------------


def _seed_inputs(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random strictly increasing non-uniform knots, evaluation x, and 3 features."""
    rng = np.random.default_rng(57200 + seed)
    n_knots = 4 + (seed % 4)  # 4..7 knots
    spacings = rng.uniform(0.15, 2.0, size=n_knots - 1)
    start = float(rng.uniform(-3.0, 2.0))
    knots = start + np.concatenate([[0.0], np.cumsum(spacings)])
    span = float(knots[-1] - knots[0])
    x_in = rng.uniform(knots[0], knots[-1], size=80)
    x_out = np.concatenate(
        [
            rng.uniform(knots[0] - 0.5 * span, knots[0], size=10),
            rng.uniform(knots[-1], knots[-1] + 0.5 * span, size=10),
        ]
    )
    xs = np.concatenate([x_in, x_out])
    feats = rng.standard_normal((100, 3))
    return knots, xs, feats


def _grid_rheon_outputs() -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    out = []
    for seed in GRID_SEEDS:
        knots, xs, feats = _seed_inputs(seed)
        k_list = knots.tolist()
        df_x = pl.DataFrame({"x": xs})
        b_const = df_x.select(
            bspline("x", k_list, degree=3, include_bias=True, extrapolate="constant")
        ).to_numpy()
        b_lin = df_x.select(
            bspline("x", k_list, degree=3, include_bias=False, extrapolate="linear")
        ).to_numpy()
        ns_mat = df_x.select(natural_spline("x", k_list, include_bias=False)).to_numpy()
        df_f = pl.DataFrame({"a": feats[:, 0], "b": feats[:, 1], "c": feats[:, 2]})
        inter_mat = df_f.select(
            interactions(["a", "b", "c"], degree=3, include_self=(seed % 2 == 1))
        ).to_numpy()
        out.append((b_const, b_lin, ns_mat, inter_mat))
    return out


ledger.guard(
    "spline_oracle",
    _grid_rheon_outputs,
    oracles=["scipy", "scikit-learn"],
    bits=52,
)


@pytest.mark.oracle_grid(
    reason="50 seeds comparing bspline, natural_spline, and interactions to scipy and scikit-learn"
)
def test_basis_operators_match_scipy_and_sklearn_over_50_seeds() -> None:
    rheon_vals = _grid_rheon_outputs()
    bs_errs, bs_lin_errs, sk_sub_errs, ns_qr_errs, ns_cs_errs, inter_errs = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for seed, (b_const, b_lin, ns_mat, inter_mat) in zip(GRID_SEEDS, rheon_vals, strict=True):
        knots, xs, feats = _seed_inputs(seed)
        rng = np.random.default_rng(99000 + seed)

        want_const = _scipy_clamped_bspline(xs, knots, 3, include_bias=True, extrapolate="constant")
        bs_errs.append(float(np.max(np.abs(b_const - want_const))))

        want_lin = _scipy_clamped_bspline(xs, knots, 3, include_bias=False, extrapolate="linear")
        bs_lin_errs.append(float(np.max(np.abs(b_lin - want_lin))))

        # Subspace parity vs sklearn SplineTransformer on interior points xs[:80].
        x_in = xs[:80]
        st = SplineTransformer(
            knots=knots[:, None], degree=3, include_bias=True, extrapolation="constant"
        ).fit(x_in[:, None])
        sk_in = st.transform(x_in[:, None])
        coef_sk, *_ = np.linalg.lstsq(b_const[:80], sk_in, rcond=None)
        sk_sub_errs.append(float(np.max(np.abs(b_const[:80] @ coef_sk - sk_in))))

        # Natural spline vs scipy BSpline + scipy.linalg.qr (both inside & linear outside).
        want_ns = _align_column_signs(
            ns_mat, _scipy_natural_spline_qr(xs, knots, include_bias=False)
        )
        ns_qr_errs.append(float(np.max(np.abs(ns_mat - want_ns))))

        # Natural spline + intercept vs scipy CubicSpline(bc_type="natural") on interior.
        y_k = rng.standard_normal(len(knots))
        y_cs = CubicSpline(knots, y_k, bc_type="natural")(x_in)
        design = np.column_stack([np.ones(len(x_in)), ns_mat[:80]])
        coef_cs, *_ = np.linalg.lstsq(design, y_cs, rcond=None)
        ns_cs_errs.append(float(np.max(np.abs(design @ coef_cs - y_cs))))

        # Interactions vs sklearn PolynomialFeatures.
        inc_self = seed % 2 == 1
        pf = PolynomialFeatures(degree=3, interaction_only=not inc_self, include_bias=False)
        want_inter = pf.fit_transform(feats)[:, 3:]
        inter_errs.append(float(np.max(np.abs(inter_mat - want_inter))))

    ledger.record("spline_oracle", "bspline_deg3_max_err", bs_errs, floor=1e-13)
    ledger.record("spline_oracle", "bspline_linear_extrap_max_err", bs_lin_errs, floor=1e-13)
    ledger.record("spline_oracle", "bspline_sklearn_subspace_err", sk_sub_errs, floor=1e-13)
    ledger.record("spline_oracle", "natural_spline_qr_max_err", ns_qr_errs, floor=1e-13)
    ledger.record("spline_oracle", "natural_spline_cubicspline_err", ns_cs_errs, floor=1e-13)
    ledger.record("spline_oracle", "interactions_poly_max_err", inter_errs, floor=1e-13)

    assert max(bs_errs) < 1e-13
    assert max(bs_lin_errs) < 1e-13
    assert max(sk_sub_errs) < 1e-12
    assert max(ns_qr_errs) < 1e-12
    assert max(ns_cs_errs) < 1e-12
    assert max(inter_errs) < 1e-13
