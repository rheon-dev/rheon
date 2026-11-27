"""Linear equality constraints (R beta_slopes = r) on ewridge, rls, and gram.solve.

Holds constrained fits to:
- statsmodels ``GLM(Gaussian).fit_constrained`` and ``RecursiveLS(constraints=)``
  over a 50-seed oracle grid recorded in the oracle ledger;
- exact KKT optimality and exact projected covariance ``se_coef``,
  ``error_inflation``, and ``support_coef``;
- exact ``rls`` <-> ``ewridge(ridge_scale="sum")`` parity (<= 1e-12);
- ``gram.solve`` (single & multi-ridge, standardized, through-origin, within);
- hard rules: chunk invariance, save/load bit-identity, singular ``R A R'``
  withholding, and eager validation refusals.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

import ledger
from rheon import gram as rg
from rheon import polars as rp

TIER = "mixed"
GRID_SEEDS = range(50)


def _linear_frame(seed: int, n: int = 400, k: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(20_516 + seed)
    x = rng.normal(size=(n, k))
    x[:, 1] += 0.4 * x[:, 0]
    x[:, 2] += -0.3 * x[:, 1]
    beta_true = np.array([0.45, 0.35, 0.20]) if k == 3 else np.full(k, 1.0 / k)
    y = 0.8 + x @ beta_true + 0.35 * rng.normal(size=n)
    cols = {f"x{j}": x[:, j] for j in range(k)}
    cols["y"] = y
    return pl.DataFrame(cols)


def _sm_glm_constrained(
    x: np.ndarray,
    y: np.ndarray,
    r_mat: np.ndarray,
    rhs: np.ndarray,
    *,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if fit_intercept:
        x_des = sm.add_constant(x, has_constant="add")
        r_full = np.hstack([np.zeros((r_mat.shape[0], 1)), r_mat])
    else:
        x_des = x
        r_full = r_mat
    res = sm.GLM(y, x_des, family=sm.families.Gaussian()).fit_constrained((r_full, rhs))
    return np.asarray(res.params, dtype=float), np.asarray(res.bse, dtype=float)


def _kkt_ridge_constrained(
    x: np.ndarray,
    y: np.ndarray,
    r_mat: np.ndarray,
    rhs: np.ndarray,
    ridge: float,
    *,
    fit_intercept: bool = True,
    ridge_scale: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Direct KKT solution for constrained ridge regression and projected A_c."""
    n, k = x.shape
    if fit_intercept:
        z = np.column_stack([np.ones(n), x])
        d = k + 1
        r_full = np.hstack([np.zeros((r_mat.shape[0], 1)), r_mat])
    else:
        z = x
        d = k
        r_full = r_mat
    q = r_mat.shape[0]
    if ridge_scale == "sum":
        h = z.T @ z + ridge * np.eye(d)
        b = z.T @ y
    else:
        h = (z.T @ z) / n
        pen = np.eye(d)
        if fit_intercept:
            pen[0, 0] = 0.0
        h = h + ridge * pen
        b = (z.T @ y) / n
    kkt = np.block([[h, r_full.T], [r_full, np.zeros((q, q))]])
    sol = np.linalg.solve(kkt, np.concatenate([b, rhs]))
    beta = sol[:d]
    h_inv = np.linalg.inv(h)
    m = r_full @ h_inv @ r_full.T
    a_c = h_inv - h_inv @ r_full.T @ np.linalg.solve(m, r_full @ h_inv)
    if ridge_scale == "mean":
        a_c = a_c / n
    return beta, a_c


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ewridge_matches_statsmodels_glm_constrained_and_rls_parity(fit_intercept: bool):
    df = _linear_frame(1, n=350, k=3)
    r_mat = np.array([[1.0, 1.0, 1.0], [1.0, -2.0, 0.0]])
    rhs = np.array([1.0, 0.1])
    x = df.select(["x0", "x1", "x2"]).to_numpy()
    y = df["y"].to_numpy()

    # Unpenalized ewridge (ridge=0, half_life=inf) matches statsmodels GLM.fit_constrained to 1e-12.
    spec_ols = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1", "x2"],
        half_life=float("inf"),
        ridge=0.0,
        fit_intercept=fit_intercept,
        constraints=(r_mat, rhs),
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
        emit_se_coef=True,
        emit_error_inflation=True,
    )
    bank_ols = rp.ModelBank([spec_ols])
    out_ols = bank_ols.fit_predict(df).unnest("m")
    got_beta = np.asarray(out_ols["coef"].drop_nulls()[-1].to_list(), dtype=float)
    sm_beta, _ = _sm_glm_constrained(x, y, r_mat, rhs, fit_intercept=fit_intercept)
    np.testing.assert_allclose(got_beta, sm_beta, rtol=1e-11, atol=1e-11)

    # Verify projected se_coef = sigma * sqrt(diag(A_c)).
    kkt_beta, a_c = _kkt_ridge_constrained(
        x, y, r_mat, rhs, 0.0, fit_intercept=fit_intercept, ridge_scale="mean"
    )
    np.testing.assert_allclose(got_beta, kkt_beta, rtol=1e-11, atol=1e-11)
    spec_sig = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1", "x2"],
        half_life=float("inf"),
        ridge=0.0,
        fit_intercept=fit_intercept,
        constraints=(r_mat, rhs),
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
        emit_sigma=True,
        emit_se_coef=True,
    )
    out_sig = rp.ModelBank([spec_sig]).fit_predict(df).unnest("m")
    sigma_last = float(out_sig["sigma_y"].drop_nulls()[-1])
    se_last = np.asarray(out_sig["se_coef"].drop_nulls()[-1].to_list(), dtype=float)
    np.testing.assert_allclose(
        se_last, sigma_last * np.sqrt(np.maximum(np.diag(a_c), 0.0)), rtol=1e-11, atol=1e-12
    )

    # Exact parity between rls and ewridge(ridge_scale="sum") under decay & weights.
    # Use matching warm-up gates (`min_weight=0.0`, no error-inflation gate) so both
    # models fold the exact same residuals into `sigma` from the same first row.
    rng = np.random.default_rng(77)
    df_w = df.with_columns(pl.Series("w", rng.uniform(0.5, 2.0, size=df.height)))
    s_ew = rp.spec.ewridge(
        "ew",
        targets=["y"],
        features=["x0", "x1", "x2"],
        half_life=45.0,
        ridge=0.5,
        ridge_scale="sum",
        fit_intercept=fit_intercept,
        weight="w",
        min_weight=0.0,
        max_error_inflation=float("inf"),
        constraints=(r_mat, rhs),
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
        emit_se_coef=True,
        emit_error_inflation=True,
    )
    s_rls = rp.spec.rls(
        "rls",
        targets=["y"],
        features=["x0", "x1", "x2"],
        half_life=45.0,
        delta=0.5,
        fit_intercept=fit_intercept,
        weight="w",
        min_weight=0.0,
        constraints=(r_mat, rhs),
        coef_every=0.0,
        emit_se_coef=True,
        emit_error_inflation=True,
    )
    both = rp.ModelBank([s_ew, s_rls]).fit_predict(df_w)
    ew = both.select("ew").unnest("ew")
    rl = both.select("rls").unnest("rls")
    # Compare after warm-up (row 20 onward).
    np.testing.assert_allclose(
        ew["pred_y"].to_numpy()[20:], rl["pred_y"].to_numpy()[20:], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        ew["error_inflation_y"].to_numpy()[20:],
        rl["error_inflation_y"].to_numpy()[20:],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ew["coef"].drop_nulls()[-1].to_list(),
        rl["coef"].drop_nulls()[-1].to_list(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ew["se_coef"].drop_nulls()[-1].to_list(),
        rl["se_coef"].drop_nulls()[-1].to_list(),
        rtol=1e-12,
        atol=1e-12,
    )
    ew_sup = np.asarray(ew["support_coef"].drop_nulls()[-1].to_list(), dtype=float)
    slope_sup = ew_sup[int(fit_intercept) :]
    if fit_intercept:
        assert np.isnan(ew_sup[0])
    assert np.all(np.isfinite(slope_sup))
    assert np.all((slope_sup >= -1e-12) & (slope_sup <= 1.0 + 1e-12))
    assert slope_sup.sum() <= (3 - r_mat.shape[0]) + 1e-9


def test_rls_matches_statsmodels_recursive_ls_constrained():
    """Undecayed RLS with small delta matches statsmodels RecursiveLS(constraints=)."""
    import pandas as pd

    df = _linear_frame(5, n=300, k=3)
    r_mat = np.array([[1.0, 1.0, 1.0]])
    rhs = np.array([1.0])
    x = df.select(["x0", "x1", "x2"]).to_numpy().copy()
    y = pd.Series(df["y"].to_numpy().copy())
    x_des = sm.add_constant(x, has_constant="add")
    r_full = np.hstack([np.zeros((1, 1)), r_mat])

    sm_rls = sm.RecursiveLS(y, x_des, constraints=(r_full, rhs)).fit()
    spec = rp.spec.rls(
        "m",
        targets=["y"],
        features=["x0", "x1", "x2"],
        half_life=float("inf"),
        delta=1e-10,
        constraints=(r_mat, rhs),
        coef_every=0.0,
    )
    out = rp.ModelBank([spec]).fit_predict(df).unnest("m")
    got = np.asarray(out["coef"].drop_nulls()[-1].to_list(), dtype=float)
    np.testing.assert_allclose(got, sm_rls.params.to_numpy(), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_gram_solve_constraints_matches_ewridge_and_kkt(standardize: bool, fit_intercept: bool):
    df = _linear_frame(9, n=250, k=3)
    r_mat = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]])
    rhs = np.array([1.0, 0.25])
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1", "x2"],
        half_life=float("inf"),
        ridge=[0.0, 0.2],
        standardize=standardize,
        fit_intercept=fit_intercept,
        constraints=(r_mat, rhs),
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    bank = rp.ModelBank([spec])
    bank.fit(df)
    g = bank.gram("m")[0]
    solved = rg.solve(
        g,
        ridge=[0.0, 0.2],
        standardize=standardize,
        constraints=(r_mat, rhs),
    )
    live = bank.coef("m")
    for idx, r_val in enumerate((0.0, 0.2)):
        sub_l = live.filter(pl.col("ridge") == r_val).sort("position")
        assert g["columns"] == sub_l["term"].to_list()
        np.testing.assert_allclose(solved[idx], sub_l["coef"].to_numpy(), rtol=1e-11, atol=1e-11)


def test_chunk_invariance_save_load_and_singular_constraint():
    df = _linear_frame(12, n=250, k=3)
    r_mat = [[1.0, 1.0, 1.0]]
    rhs = [1.0]
    for builder, extra in (
        (rp.spec.ewridge, {"ridge": 0.05, "solve_every": 0.0, "max_rows_between_solves": 1}),
        (rp.spec.rls, {"delta": 0.5}),
    ):
        spec = builder(
            "m",
            targets=["y"],
            features=["x0", "x1", "x2"],
            half_life=40.0,
            constraints=(r_mat, rhs),
            coef_every=0.0,
            emit_se_coef=True,
            emit_error_inflation=True,
            **extra,
        )
        whole = rp.ModelBank([spec]).fit_predict(df).unnest("m")
        b_c = rp.ModelBank([spec])
        chunked = pl.concat(
            [b_c.fit_predict(df.slice(i, 29)) for i in range(0, df.height, 29)]
        ).unnest("m")
        cols = ["pred_y", "coef", "se_coef", "error_inflation_y"]
        if "support_coef" in whole.columns:
            cols.append("support_coef")
        for col in cols:
            assert chunked[col].equals(whole[col], null_equal=True), col

        b1 = rp.ModelBank([spec])
        p1 = b1.fit_predict(df.slice(0, 140))
        b2 = rp.ModelBank.load_bytes(b1.save_bytes())
        p2 = b2.fit_predict(df.slice(140))
        resumed = pl.concat([p1, p2]).unnest("m")
        for col in cols:
            assert resumed[col].equals(whole[col], null_equal=True), col

    # Constant constrained column under standardize=True -> R A R' singular -> withheld!
    df_const = df.with_columns(pl.lit(3.0).alias("x0"))
    spec_sing = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1", "x2"],
        half_life=float("inf"),
        standardize=True,
        constraints=([[1.0, 0.0, 0.0]], [1.0]),
        solve_every=0.0,
        max_rows_between_solves=1,
    )
    bank_sing = rp.ModelBank([spec_sing])
    out_sing = bank_sing.fit_predict(df_const).unnest("m")
    assert out_sing["pred_y"].null_count() == df_const.height


def test_constraints_validation_refusals():
    base = dict(targets=["y"], features=["x0", "x1", "x2"], half_life=50.0)
    # Empty constraint matrix.
    with pytest.raises(ValueError, match="at least 1 row"):
        rp.spec.ewridge("m", **base, constraints=([], []))
    # Shape mismatch.
    with pytest.raises(ValueError, match="must have 3 entries"):
        rp.spec.ewridge("m", **base, constraints=([[1.0, 1.0]], [1.0]))
    with pytest.raises(ValueError, match="one entry per row"):
        rp.spec.ewridge("m", **base, constraints=([[1.0, 1.0, 1.0]], [1.0, 2.0]))
    # Rank-deficient rows.
    with pytest.raises(ValueError, match="full row rank"):
        rp.spec.ewridge("m", **base, constraints=([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0]], [1.0, 2.0]))
    # Zero row.
    with pytest.raises(ValueError, match="zero"):
        rp.spec.rls("m", **base, constraints=([[0.0, 0.0, 0.0]], [0.0]))
    # q > k.
    with pytest.raises(ValueError, match="at most"):
        rp.spec.rls(
            "m",
            **base,
            constraints=(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                [0.0, 0.0, 0.0, 1.0],
            ),
        )
    # Refused beside feature_sets.
    with pytest.raises(ValueError, match="feature_sets"):
        rp.spec.ewridge(
            "m",
            **base,
            feature_sets={"a": ["x0", "x1"]},
            constraints=([[1.0, 1.0, 1.0]], [1.0]),
        )
    # Also validated in gram.solve.
    df = _linear_frame(0, n=50, k=3)
    bank = rp.ModelBank([rp.spec.ewridge("m", **base)])
    bank.fit(df)
    g = bank.gram("m")[0]
    with pytest.raises(ValueError, match="full row rank"):
        rg.solve(g, constraints=([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0]], [1.0, 2.0]))


# ---------------------------------------------------------------------------
# 50-seed oracle grid + ledger guard against statsmodels GLM.fit_constrained
# ---------------------------------------------------------------------------

_GRID_R = np.array([[1.0, 1.0, 1.0], [1.0, -2.0, 0.0]])
_GRID_RHS = np.array([1.0, 0.15])


def _grid_rheon_outputs() -> list[tuple[list[float], list[float], list[float]]]:
    out = []
    for seed in GRID_SEEDS:
        df = _linear_frame(seed, n=400, k=3)
        s_ew = rp.spec.ewridge(
            "ew",
            targets=["y"],
            features=["x0", "x1", "x2"],
            half_life=float("inf"),
            ridge=0.0,
            constraints=(_GRID_R, _GRID_RHS),
            solve_every=0.0,
            max_rows_between_solves=1,
            coef_every=0.0,
        )
        s_rls = rp.spec.rls(
            "rls",
            targets=["y"],
            features=["x0", "x1", "x2"],
            half_life=float("inf"),
            delta=1e-10,
            constraints=(_GRID_R, _GRID_RHS),
            coef_every=0.0,
        )
        bank = rp.ModelBank([s_ew, s_rls])
        res = bank.fit_predict(df)
        ew_beta = res.select("ew").unnest("ew")["coef"].drop_nulls()[-1].to_list()
        rls_beta = res.select("rls").unnest("rls")["coef"].drop_nulls()[-1].to_list()
        g_beta = rg.solve(bank.gram("ew")[0], ridge=0.0, constraints=(_GRID_R, _GRID_RHS)).tolist()
        out.append((ew_beta, rls_beta, g_beta))
    return out


ledger.guard(
    "linear_constraints_oracle",
    _grid_rheon_outputs,
    oracles=["statsmodels"],
    bits=52,
)


@pytest.mark.oracle_grid(
    reason="50 seeds of 400 rows comparing constrained ewridge, rls, and gram.solve to statsmodels"
)
def test_linear_constraints_match_statsmodels_over_50_seeds():
    rheon_vals = _grid_rheon_outputs()
    ew_errs, rls_errs, gram_errs = [], [], []
    for seed, (ew_b, rls_b, g_b) in zip(GRID_SEEDS, rheon_vals, strict=True):
        df = _linear_frame(seed, n=400, k=3)
        x = df.select(["x0", "x1", "x2"]).to_numpy()
        y = df["y"].to_numpy()
        want, _ = _sm_glm_constrained(x, y, _GRID_R, _GRID_RHS, fit_intercept=True)
        scale = float(np.linalg.norm(want))
        ew_errs.append(float(np.linalg.norm(np.asarray(ew_b) - want)) / scale)
        rls_errs.append(float(np.linalg.norm(np.asarray(rls_b) - want)) / scale)
        gram_errs.append(float(np.linalg.norm(np.asarray(g_b) - want)) / scale)

    ledger.record("linear_constraints_oracle", "ewridge_rel_err", ew_errs, floor=1e-12)
    ledger.record("linear_constraints_oracle", "rls_rel_err", rls_errs, floor=1e-8)
    ledger.record("linear_constraints_oracle", "gram_solve_rel_err", gram_errs, floor=1e-12)

    assert max(ew_errs) < 1e-11
    assert max(rls_errs) < 1e-7
    assert max(gram_errs) < 1e-11
