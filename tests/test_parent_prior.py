"""Tests and 50-seed SciPy oracles for hierarchical parent priors on ``ewridge``."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
import pytest
from scipy.linalg import solve as scipy_solve
from scipy.optimize import Bounds, LinearConstraint, minimize

import ledger
import rheon.polars as rp

TIER = "mixed"
GRID_SEEDS = range(50)


def _decay_weights(times: np.ndarray, weights: np.ndarray, half_life: float) -> np.ndarray:
    """Exact exponentially decayed row weights at the end of a stream."""
    if math.isinf(half_life):
        return weights.astype(float).copy()
    dt = times[-1] - times
    return weights.astype(float) * np.power(0.5, dt / half_life)


def _scipy_hierarchical_ridge(
    x: np.ndarray,
    y: np.ndarray,
    row_w: np.ndarray,
    beta_parent: np.ndarray,
    *,
    ridge: float,
    prior_weight: float,
    standardize: bool,
    fit_intercept: bool,
    prior_include_intercept: bool,
    ridge_scale: str = "mean",
    prior_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Independent closed-form SciPy oracle directly from raw rows.

    Returns ``(beta, cov_per_noise, edf, support_coef)`` in original feature
    units without invoking any ``rheon`` code.
    """
    n, k = x.shape
    w_sum = float(np.sum(row_w))
    n_kish = (w_sum * w_sum) / float(np.sum(row_w * row_w))
    w_norm = row_w / w_sum

    if ridge_scale == "sum":
        z = np.column_stack([np.ones(n), x]) if fit_intercept else x
        d = z.shape[1]
        gram = (z.T * row_w) @ z
        d_prior = np.ones(d)
        if fit_intercept and not prior_include_intercept:
            d_prior[0] = 0.0
        pen = prior_scale * ridge * np.ones(d) + prior_weight * d_prior
        h = gram + np.diag(pen)
        rhs = z.T @ (row_w * y) + prior_weight * (d_prior * beta_parent)
        beta = scipy_solve(h, rhs, assume_a="pos")
        h_inv = scipy_solve(h, np.eye(d), assume_a="pos")
        edf = float(np.trace(h_inv @ gram))
        cov_per_noise = (w_sum / n_kish) * h_inv
        support = 1.0 - pen * np.diag(h_inv)
        return beta, cov_per_noise, edf, support

    if fit_intercept:
        mx = w_norm @ x
        xc = x - mx
        if standardize:
            # Standardized centered coordinates: z_tilde = [1, (x - mx) / s]
            v = np.sum((xc * xc) * w_norm[:, None], axis=0)
            s = np.sqrt(v)
            z_tilde = np.column_stack([np.ones(n), xc / s])
            gram = (z_tilde.T * row_w) @ z_tilde
            d_ridge = np.concatenate([[0.0], np.full(k, w_sum * ridge)])
            d_prior = np.concatenate(
                [[prior_weight if prior_include_intercept else 0.0], np.full(k, prior_weight)]
            )
            gamma_p = np.concatenate(
                [[beta_parent[0] + float(mx @ beta_parent[1:])], s * beta_parent[1:]]
            )
            h = gram + np.diag(d_ridge + d_prior)
            rhs = z_tilde.T @ (row_w * y) + d_prior * gamma_p
            gamma = scipy_solve(h, rhs, assume_a="pos")
            h_inv = scipy_solve(h, np.eye(k + 1), assume_a="pos")
            edf = float(np.trace(h_inv @ gram))

            slopes = gamma[1:] / s
            intercept = float(gamma[0] - mx @ slopes)
            beta = np.concatenate([[intercept], slopes])

            # Transform covariance back to original units: beta = J @ gamma
            j_mat = np.zeros((k + 1, k + 1))
            j_mat[0, 0] = 1.0
            j_mat[0, 1:] = -mx / s
            j_mat[1:, 1:] = np.diag(1.0 / s)
            cov_per_noise = (w_sum / n_kish) * (j_mat @ h_inv @ j_mat.T)

            # Support matches centred ewridge: NaN on intercept, 1 - shift * inv_diag on slopes
            pen_slopes = w_sum * ridge + prior_weight
            support = np.concatenate([[math.nan], 1.0 - pen_slopes * np.diag(h_inv)[1:]])
            return beta, cov_per_noise, edf, support

        # Unstandardized with intercept: z = [1, x]
        z = np.column_stack([np.ones(n), x])
        gram = (z.T * row_w) @ z
        d_ridge = np.concatenate([[0.0], np.full(k, w_sum * ridge)])
        d_prior = np.concatenate(
            [[prior_weight if prior_include_intercept else 0.0], np.full(k, prior_weight)]
        )
        h = gram + np.diag(d_ridge + d_prior)
        rhs = z.T @ (row_w * y) + d_prior * beta_parent
        beta = scipy_solve(h, rhs, assume_a="pos")
        h_inv = scipy_solve(h, np.eye(k + 1), assume_a="pos")
        edf = float(np.trace(h_inv @ gram))
        cov_per_noise = (w_sum / n_kish) * h_inv
        pen_slopes = w_sum * ridge + prior_weight
        support = np.concatenate([[math.nan], 1.0 - pen_slopes * np.diag(h_inv)[1:]])
        return beta, cov_per_noise, edf, support

    # Through-origin (fit_intercept=False)
    if standardize:
        s = np.sqrt(np.sum((x * x) * w_norm[:, None], axis=0))
        z_tilde = x / s
        gram = (z_tilde.T * row_w) @ z_tilde
        pen = w_sum * ridge + prior_weight
        h = gram + pen * np.eye(k)
        rhs = z_tilde.T @ (row_w * y) + prior_weight * (s * beta_parent)
        gamma = scipy_solve(h, rhs, assume_a="pos")
        h_inv = scipy_solve(h, np.eye(k), assume_a="pos")
        edf = float(np.trace(h_inv @ gram))
        beta = gamma / s
        inv_s = np.diag(1.0 / s)
        cov_per_noise = (w_sum / n_kish) * (inv_s @ h_inv @ inv_s)
        support = 1.0 - pen * np.diag(h_inv)
        return beta, cov_per_noise, edf, support

    gram = (x.T * row_w) @ x
    pen = w_sum * ridge + prior_weight
    h = gram + pen * np.eye(k)
    rhs = x.T @ (row_w * y) + prior_weight * beta_parent
    beta = scipy_solve(h, rhs, assume_a="pos")
    h_inv = scipy_solve(h, np.eye(k), assume_a="pos")
    edf = float(np.trace(h_inv @ gram))
    cov_per_noise = (w_sum / n_kish) * h_inv
    support = 1.0 - pen * np.diag(h_inv)
    return beta, cov_per_noise, edf, support


def _seed_problem(seed: int) -> dict[str, Any]:
    """Deterministic multi-group hierarchical problem for one oracle-grid seed."""
    rng = np.random.default_rng(20_000 + seed)
    k = [2, 3, 4, 5][seed % 4]
    n_groups = 3
    n = 90 + (seed % 4) * 20
    fit_intercept = (seed % 2) == 0
    standardize = ((seed // 2) % 2) == 1
    prior_include_intercept = True if not fit_intercept else ((seed // 4) % 2) == 0
    half_life = [math.inf, 25.0, 50.0][seed % 3]
    ridge = [0.0, 1e-4, 0.05, 0.4][seed % 4]
    prior_weight = [0.5, 4.0, 15.0, 80.0][(seed // 3) % 4]

    scales = np.power(10.0, rng.uniform(-1.5, 1.5, size=k))
    means = rng.uniform(-3.0, 3.0, size=k)
    a_gen = rng.standard_normal((k, k))
    x = (rng.standard_normal((n, k)) @ a_gen / math.sqrt(k)) * scales + means
    beta_global = rng.uniform(-1.5, 1.5, size=k) / scales
    alpha_global = 1.2 if fit_intercept else 0.0

    # Three groups with imbalanced counts (roughly 60%, 28%, 12%) and group offsets.
    groups = rng.choice(["g0", "g1", "g2"], size=n, p=[0.6, 0.28, 0.12])
    # Ensure every group appears at least k + 3 times so every group has finite sigma & fit.
    for gi, gname in enumerate(["g0", "g1", "g2"]):
        idx = np.where(groups == gname)[0]
        if len(idx) < k + 4:
            groups[gi * (k + 4) : (gi + 1) * (k + 4)] = gname

    group_deltas = {
        "g0": (0.2 if fit_intercept else 0.0, rng.normal(0.0, 0.25, size=k) / scales),
        "g1": (-0.3 if fit_intercept else 0.0, rng.normal(0.0, 0.35, size=k) / scales),
        "g2": (0.5 if fit_intercept else 0.0, rng.normal(0.0, 0.5, size=k) / scales),
    }
    y = np.empty(n, dtype=float)
    for i in range(n):
        da, db = group_deltas[str(groups[i])]
        y[i] = (alpha_global + da) + float(x[i] @ (beta_global + db)) + 0.3 * rng.standard_normal()

    times = np.cumsum(rng.uniform(0.5, 2.0, size=n))
    weights = rng.uniform(0.4, 2.5, size=n)

    return {
        "k": k,
        "n": n,
        "n_groups": n_groups,
        "fit_intercept": fit_intercept,
        "standardize": standardize,
        "prior_include_intercept": prior_include_intercept,
        "half_life": half_life,
        "ridge": ridge,
        "prior_weight": prior_weight,
        "x": x,
        "y": y,
        "groups": groups.tolist(),
        "times": times,
        "weights": weights,
    }


def _grid_rheon_outputs() -> list[dict[str, Any]]:
    """Run rheon on all 50 seeds (both ``prior_from`` and column ``coef_prior``)."""
    out: list[dict[str, Any]] = []
    for seed in GRID_SEEDS:
        p = _seed_problem(seed)
        k = p["k"]
        feat_names = [f"x{i}" for i in range(k)]
        df = pl.DataFrame(
            {
                "t": p["times"],
                "w": p["weights"],
                "g": p["groups"],
                "y": p["y"],
                **{feat_names[i]: p["x"][:, i] for i in range(k)},
            }
        )
        parent = rp.spec.ewridge(
            "parent",
            targets=["y"],
            features=feat_names,
            clock="t",
            weight="w",
            half_life=p["half_life"],
            gap_cap=1e6,
            ridge=max(p["ridge"], 1e-4),
            standardize=p["standardize"],
            fit_intercept=p["fit_intercept"],
            solve_every=0.0,
            max_rows_between_solves=1,
            coef_every=0.0,
            max_error_inflation=float("inf"),
        )
        child = rp.spec.ewridge(
            "child",
            targets=["y"],
            features=feat_names,
            clock="t",
            weight="w",
            group="g",
            half_life=p["half_life"],
            gap_cap=1e6,
            ridge=p["ridge"],
            standardize=p["standardize"],
            fit_intercept=p["fit_intercept"],
            prior_from="parent",
            prior_weight=p["prior_weight"],
            prior_include_intercept=p["prior_include_intercept"],
            solve_every=0.0,
            max_rows_between_solves=1,
            coef_every=0.0,
            emit_sigma=True,
            emit_se_coef=True,
            emit_error_inflation=True,
            max_error_inflation=float("inf"),
        )
        bank = rp.ModelBank([parent, child])
        res = bank.fit_predict(df)

        # Also run dynamic column mode using the parent's emitted coef column.
        df_col = df.with_columns(res["parent"].struct.field("coef").alias("p_coef"))
        child_col = rp.spec.ewridge(
            "child_col",
            targets=["y"],
            features=feat_names,
            clock="t",
            weight="w",
            group="g",
            half_life=p["half_life"],
            gap_cap=1e6,
            ridge=p["ridge"],
            standardize=p["standardize"],
            fit_intercept=p["fit_intercept"],
            coef_prior="p_coef",
            prior_weight=p["prior_weight"],
            prior_include_intercept=p["prior_include_intercept"],
            solve_every=0.0,
            max_rows_between_solves=1,
            coef_every=0.0,
            max_error_inflation=float("inf"),
        )
        bank_col = rp.ModelBank([child_col])
        res_col = bank_col.fit_predict(df_col)

        # Grab final row per group for child, parent coef at that exact row, and diagnostics.
        per_group: dict[str, dict[str, Any]] = {}
        g_col = p["groups"]
        child_s = res["child"]
        parent_s = res["parent"]
        col_s = res_col["child_col"]
        for gname in ("g0", "g1", "g2"):
            last_idx = max(i for i, gv in enumerate(g_col) if gv == gname)
            per_group[gname] = {
                "last_idx": last_idx,
                "parent_coef": parent_s.struct.field("coef")[last_idx].to_list(),
                "child_coef": child_s.struct.field("coef")[last_idx].to_list(),
                "col_coef": col_s.struct.field("coef")[last_idx].to_list(),
                "support_coef": child_s.struct.field("support_coef")[last_idx].to_list(),
                "se_coef": child_s.struct.field("se_coef")[last_idx].to_list(),
                "sigma": float(child_s.struct.field("sigma_y")[last_idx]),
                "error_inflation": float(child_s.struct.field("error_inflation_y")[last_idx]),
            }
        # Also read post-fit coefficients from bank.coef("child") for each group.
        coef_tbl = bank.coef("child")
        for gname in ("g0", "g1", "g2"):
            sub = coef_tbl.filter(pl.col("group") == gname)
            per_group[gname]["final_coef"] = sub["coef"].to_list()
        out.append(per_group)
    return out


ledger.guard(
    "parent_prior_oracle",
    _grid_rheon_outputs,
    oracles=["scipy"],
)


@pytest.mark.oracle_grid(
    reason="50 seeds comparing hierarchical parent-prior ewridge to closed-form SciPy linear solve"
)
def test_parent_prior_50_seed_scipy_oracle() -> None:
    """50-seed independent SciPy oracle for ``prior_from`` and column ``coef_prior``."""
    rheon_vals = _grid_rheon_outputs()
    coef_rel_errs: list[float] = []
    se_rel_errs: list[float] = []
    diag_rel_errs: list[float] = []

    for seed, per_group in zip(GRID_SEEDS, rheon_vals, strict=True):
        p = _seed_problem(seed)
        x = p["x"]
        y = p["y"]
        times = p["times"]
        weights = p["weights"]
        g_arr = np.asarray(p["groups"])

        seed_coef_err = 0.0
        seed_se_err = 0.0
        seed_diag_err = 0.0

        for gname in ("g0", "g1", "g2"):
            rec = per_group[gname]
            got_beta = np.asarray(rec["child_coef"], dtype=np.float64)
            got_col = np.asarray(rec["col_coef"], dtype=np.float64)
            got_final = np.asarray(rec["final_coef"], dtype=np.float64)
            beta_parent = np.asarray(rec["parent_coef"], dtype=np.float64)
            got_sup = np.asarray(rec["support_coef"], dtype=np.float64)

            # 1. Exact bit-identity between prior_from, column coef_prior, and bank.coef()
            assert np.array_equal(got_beta.view(np.uint64), got_col.view(np.uint64)), (
                f"seed {seed} {gname}: prior_from vs column coef_prior bit mismatch"
            )
            assert np.array_equal(got_beta.view(np.uint64), got_final.view(np.uint64)), (
                f"seed {seed} {gname}: emitted coef vs bank.coef() bit mismatch"
            )

            # 2. Exact closed-form SciPy oracle over this group's rows up to last_idx
            mask = g_arr == gname
            row_w = _decay_weights(times[mask], weights[mask], p["half_life"])
            ref_beta, ref_var_pn, ref_err_inf, ref_sup = _scipy_hierarchical_ridge(
                x[mask],
                y[mask],
                row_w,
                beta_parent,
                ridge=p["ridge"],
                prior_weight=p["prior_weight"],
                standardize=p["standardize"],
                fit_intercept=p["fit_intercept"],
                prior_include_intercept=p["prior_include_intercept"],
            )

            c_err = float(
                np.linalg.norm(got_beta - ref_beta) / max(1.0, float(np.linalg.norm(ref_beta)))
            )
            seed_coef_err = max(seed_coef_err, c_err)

            # Compare support_coef on finite entries (intercept is NaN when fitted).
            fin = np.isfinite(ref_sup)
            assert np.array_equal(np.isnan(got_sup), np.isnan(ref_sup))
            s_err = float(np.max(np.abs(got_sup[fin] - ref_sup[fin])))
            seed_diag_err = max(seed_diag_err, s_err)

            # NOTE: emitted se_coef and error_inflation at row `last_idx` are PRE-row-`last_idx`
            # (computed from state after the group's penultimate row!), so check them against
            # the SciPy oracle on the group's prefix excluding its last row!
            idx_g = np.where(mask)[0]
            prev_idx = idx_g[:-1]
            # Parent's coef at the group's penultimate row drove child's state before `last_idx`.
            # Re-check error_inflation and se_coef via a 1-row predict step after the full fit
            # instead: at `last_idx`, `se_coef` used `prev_parent_coef`, whereas on a fresh
            # predict row `t_next = times[-1]`, both `se_coef` and `error_inflation` correspond
            # to the exact post-`last_idx` state `ref_var_pn` and `ref_err_inf`!
            # We verify that directly below via `predict()` on a probe row.
            _ = (prev_idx, ref_var_pn, ref_err_inf)

        # Verify post-fit `se_coef` and `error_inflation` via 1-row predict probe per group!
        feat_names = [f"x{i}" for i in range(p["k"])]
        df = pl.DataFrame(
            {
                "t": p["times"],
                "w": p["weights"],
                "g": p["groups"],
                "y": p["y"],
                **{feat_names[i]: p["x"][:, i] for i in range(p["k"])},
            }
        )
        parent = rp.spec.ewridge(
            "parent",
            targets=["y"],
            features=feat_names,
            clock="t",
            weight="w",
            half_life=p["half_life"],
            gap_cap=1e6,
            ridge=max(p["ridge"], 1e-4),
            standardize=p["standardize"],
            fit_intercept=p["fit_intercept"],
            solve_every=0.0,
            max_rows_between_solves=1,
            max_error_inflation=float("inf"),
        )
        child = rp.spec.ewridge(
            "child",
            targets=["y"],
            features=feat_names,
            clock="t",
            weight="w",
            group="g",
            half_life=p["half_life"],
            gap_cap=1e6,
            ridge=p["ridge"],
            standardize=p["standardize"],
            fit_intercept=p["fit_intercept"],
            prior_from="parent",
            prior_weight=p["prior_weight"],
            prior_include_intercept=p["prior_include_intercept"],
            solve_every=0.0,
            max_rows_between_solves=1,
            emit_sigma=True,
            emit_se_coef=True,
            emit_error_inflation=True,
            max_error_inflation=float("inf"),
        )
        bank = rp.ModelBank([parent, child])
        bank.fit(df)
        probe_x = np.vstack([x[g_arr == gn][-1] for gn in ("g0", "g1", "g2")])
        probe = pl.DataFrame(
            {
                "t": [float(p["times"][-1]) + 1.0] * 3,
                "w": [1.0, 1.0, 1.0],
                "g": ["g0", "g1", "g2"],
                "y": [0.0, 0.0, 0.0],
                **{feat_names[i]: probe_x[:, i].tolist() for i in range(p["k"])},
            }
        )
        pr = bank.predict(probe)["child"]
        for gi, gname in enumerate(("g0", "g1", "g2")):
            mask = g_arr == gname
            row_w = _decay_weights(times[mask], weights[mask], p["half_life"])
            beta_parent = np.asarray(per_group[gname]["parent_coef"], dtype=np.float64)
            _, ref_cov_pn, _, _ = _scipy_hierarchical_ridge(
                x[mask],
                y[mask],
                row_w,
                beta_parent,
                ridge=p["ridge"],
                prior_weight=p["prior_weight"],
                standardize=p["standardize"],
                fit_intercept=p["fit_intercept"],
                prior_include_intercept=p["prior_include_intercept"],
            )
            got_err_inf = float(pr.struct.field("error_inflation_y")[gi])
            got_sigma = float(pr.struct.field("sigma_y")[gi])
            got_se = np.asarray(pr.struct.field("se_coef")[gi].to_list(), dtype=np.float64)
            ref_se = got_sigma * np.sqrt(np.diag(ref_cov_pn))
            z_probe = np.concatenate([[1.0], probe_x[gi]]) if p["fit_intercept"] else probe_x[gi]
            ref_row_ei = math.sqrt(1.0 + max(0.0, float(z_probe @ ref_cov_pn @ z_probe)))
            ei_err = abs(got_err_inf - ref_row_ei) / max(1.0, ref_row_ei)
            se_err = float(
                np.linalg.norm(got_se - ref_se) / max(1.0, float(np.linalg.norm(ref_se)))
            )
            seed_diag_err = max(seed_diag_err, ei_err)
            seed_se_err = max(seed_se_err, se_err)

        coef_rel_errs.append(seed_coef_err)
        se_rel_errs.append(seed_se_err)
        diag_rel_errs.append(seed_diag_err)

    ledger.record("parent_prior_oracle", "coef_rel_err", coef_rel_errs, floor=1e-10)
    ledger.record("parent_prior_oracle", "se_coef_rel_err", se_rel_errs, floor=1e-10)
    ledger.record("parent_prior_oracle", "diag_rel_err", diag_rel_errs, floor=1e-10)
    assert max(coef_rel_errs) < 1e-10, f"worst coef rel err = {max(coef_rel_errs)}"
    assert max(se_rel_errs) < 1e-10, f"worst se_coef rel err = {max(se_rel_errs)}"
    assert max(diag_rel_errs) < 1e-10, f"worst diag rel err = {max(diag_rel_errs)}"


def test_parent_prior_zero_lookahead_exact_bits() -> None:
    """Perturbing ``y[t0]`` moves NEITHER parent nor child ``pred[<= t0]`` by a single bit."""
    rng = np.random.default_rng(101)
    n = 80
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    y = 0.5 + 1.4 * x0 - 0.8 * x1 + 0.2 * rng.standard_normal(n)
    # Introduce a brand-new cold-start group "cold" at row 50 to test cold-start causality too!
    g = ["a" if i % 2 == 0 else "b" for i in range(n)]
    g[50] = "cold"
    g[51] = "cold"

    df_base = pl.DataFrame({"t": np.arange(n, dtype=float), "g": g, "x0": x0, "x1": x1, "y": y})

    for t0 in (30, 50):  # t0=30 is warm group; t0=50 is cold-start W=0 row!
        y_pert = y.copy()
        y_pert[t0] += 1e6
        df_pert = df_base.with_columns(pl.Series("y", y_pert))

        p_spec = rp.spec.ewridge(
            "p",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=1000.0,
            half_life=30.0,
            ridge=0.05,
            solve_every=0.0,
            max_rows_between_solves=1,
            coef_every=0.0,
            max_error_inflation=float("inf"),
        )
        c_spec = rp.spec.ewridge(
            "c",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=1000.0,
            group="g",
            half_life=30.0,
            ridge=0.05,
            prior_from="p",
            prior_weight=5.0,
            solve_every=0.0,
            max_rows_between_solves=1,
            coef_every=0.0,
            max_error_inflation=float("inf"),
        )
        out_base = rp.ModelBank([p_spec, c_spec]).fit_predict(df_base)
        out_pert = rp.ModelBank([p_spec, c_spec]).fit_predict(df_pert)

        for name in ("p", "c"):
            pb = out_base[name].struct.field("pred_y").to_numpy()
            pp = out_pert[name].struct.field("pred_y").to_numpy()
            # Strictly identical bit-for-bit on ALL rows <= t0 (including t0 itself!)
            assert np.array_equal(pb[: t0 + 1].view(np.uint64), pp[: t0 + 1].view(np.uint64)), (
                f"t0={t0} spec={name}: lookahead detected at row <= {t0}"
            )
            # Perturbation MUST take effect strictly after t0
            assert not np.array_equal(pb[t0 + 1 :].view(np.uint64), pp[t0 + 1 :].view(np.uint64)), (
                f"t0={t0} spec={name}: perturbation had no effect after {t0}"
            )

        # Also test dynamic column mode with unshifted parent `coef` column:
        # even though `p_coef[t0]` already reflects `y_pert[t0]`, `child_col`'s
        # `pred_y[t0]` MUST NOT see `p_coef[t0]`!
        df_col_base = df_base.with_columns(out_base["p"].struct.field("coef").alias("p_coef"))
        df_col_pert = df_pert.with_columns(out_pert["p"].struct.field("coef").alias("p_coef"))
        col_spec = rp.spec.ewridge(
            "cc",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=1000.0,
            group="g",
            half_life=30.0,
            ridge=0.05,
            coef_prior="p_coef",
            prior_weight=5.0,
            solve_every=0.0,
            max_rows_between_solves=1,
            max_error_inflation=float("inf"),
        )
        cb = (
            rp.ModelBank([col_spec])
            .fit_predict(df_col_base)["cc"]
            .struct.field("pred_y")
            .to_numpy()
        )
        cp = (
            rp.ModelBank([col_spec])
            .fit_predict(df_col_pert)["cc"]
            .struct.field("pred_y")
            .to_numpy()
        )
        assert np.array_equal(cb[: t0 + 1].view(np.uint64), cp[: t0 + 1].view(np.uint64)), (
            f"t0={t0} column mode: lookahead detected at row <= {t0}"
        )


def test_parent_prior_chunk_save_load_and_lazy_invariance() -> None:
    """1-chunk == 1-row chunks == random chunks with msgpack & JSON save/load == LazyFrame."""
    rng = np.random.default_rng(202)
    n = 75
    df = pl.DataFrame(
        {
            "t": np.cumsum(rng.uniform(0.5, 2.0, size=n)),
            "g": rng.choice(["a", "b", "c"], size=n).tolist(),
            "w": rng.uniform(0.5, 2.0, size=n),
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "y": rng.standard_normal(n),
        }
    )
    specs = [
        rp.spec.ewridge(
            "p",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            weight="w",
            half_life=20.0,
            gap_cap=50.0,
            ridge=0.05,
            standardize=True,
            coef_every=0.0,
        ),
        rp.spec.ewridge(
            "c_all",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            weight="w",
            group="g",
            half_life=20.0,
            gap_cap=50.0,
            ridge=0.05,
            standardize=True,
            prior_from="p",
            prior_weight=6.0,
            coef_every=0.0,
            emit_se_coef=True,
            emit_error_inflation=True,
        ),
        rp.spec.ewridge(
            "c_slopes",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            weight="w",
            group="g",
            half_life=20.0,
            gap_cap=50.0,
            ridge=0.05,
            prior_from="p",
            prior_weight=4.0,
            prior_include_intercept=False,
            coef_every=0.0,
        ),
    ]

    full = rp.ModelBank(specs).fit_predict(df)

    # 1-row chunks
    b1 = rp.ModelBank(specs)
    one_row = pl.concat([b1.fit_predict(df.slice(i, 1)) for i in range(n)])
    assert full.equals(one_row)

    # Random chunks with both msgpack and JSON save/load mid-stream
    b_mid = rp.ModelBank(specs)
    splits = [0, 5, 19, 34, 52, n]
    parts = []
    for idx, (a, b) in enumerate(zip(splits[:-1], splits[1:], strict=True)):
        parts.append(b_mid.fit_predict(df.slice(a, b - a)))
        if idx == 1:
            b_mid = rp.ModelBank.load_bytes(b_mid.save_bytes())
        elif idx == 3:
            # write_json() internally verifies exact byte-for-byte JSON -> msgpack faithfulness
            js = b_mid.write_json()
            assert '"prior_from"' in js and '"dynamic_prior"' in js
            b_mid = rp.ModelBank.load_bytes(b_mid.save_bytes())
    chunked = pl.concat(parts)
    assert full.equals(chunked)

    # LazyFrame streaming
    lazy_out = df.lazy().rheon.fit_predict(specs).collect()
    for name in ("p", "c_all", "c_slopes"):
        assert full[name].equals(lazy_out[name])


def test_parent_prior_zero_weight_rule9() -> None:
    """Zero-weight poison rows leave real rows bit-identical (Hard Rule 9)."""
    rng = np.random.default_rng(303)
    n = 60
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    y = 0.3 + 1.2 * x0 - 0.7 * x1 + 0.2 * rng.standard_normal(n)
    t = np.cumsum(rng.uniform(0.5, 2.0, size=n))
    w = rng.uniform(0.5, 2.0, size=n)
    g = ["a" if i % 2 == 0 else "b" for i in range(n)]

    df_clean = pl.DataFrame({"t": t, "g": g, "w": w, "x0": x0, "x1": x1, "y": y})
    rows = []
    for i in range(n):
        rows.append(
            {
                "t": float(t[i]),
                "g": g[i],
                "w": float(w[i]),
                "x0": float(x0[i]),
                "x1": float(x1[i]),
                "y": float(y[i]),
                "real": True,
            }
        )
        if i % 5 == 2:
            rows.append(
                {
                    "t": float(t[i]),
                    "g": g[i],
                    "w": 0.0,
                    "x0": 1e6,
                    "x1": -1e6,
                    "y": 1e9,
                    "real": False,
                }
            )
    df_poison = pl.DataFrame(rows)

    specs = [
        rp.spec.ewridge(
            "p",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=1000.0,
            weight="w",
            half_life=25.0,
            ridge=0.05,
            solve_every=0.0,
            max_rows_between_solves=1,
            coef_every=0.0,
        ),
        rp.spec.ewridge(
            "c",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=1000.0,
            weight="w",
            group="g",
            half_life=25.0,
            ridge=0.05,
            prior_from="p",
            prior_weight=5.0,
            solve_every=0.0,
            max_rows_between_solves=1,
            coef_every=0.0,
        ),
    ]
    out_clean = rp.ModelBank(specs).fit_predict(df_clean)
    out_poison = rp.ModelBank(specs).fit_predict(df_poison).filter(pl.col("real")).drop("real")
    assert out_clean["p"].equals(out_poison["p"])
    assert out_clean["c"].equals(out_poison["c"])


def test_parent_prior_cold_start_limits_and_predict() -> None:
    """Cold-start borrowing, W -> 0 / W >> tau limits, warming parent, and read-only predict."""
    rng = np.random.default_rng(404)
    n = 120
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x0 - 1.5 * x1 + 0.1 * rng.standard_normal(n)
    g = ["main"] * 100 + ["new_grp"] * 20
    # Give new_grp a distinct true slope so borrowing vs own-data limits are unmistakable.
    y[100:] = -2.0 - 1.0 * x0[100:] + 3.0 * x1[100:] + 0.1 * rng.standard_normal(20)
    df = pl.DataFrame({"t": np.arange(n, dtype=float), "g": g, "x0": x0, "x1": x1, "y": y})

    p_spec = rp.spec.ewridge(
        "p",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        half_life=math.inf,
        ridge=1e-4,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    c_tight = rp.spec.ewridge(
        "c_tight",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        group="g",
        half_life=math.inf,
        ridge=1e-4,
        prior_from="p",
        prior_weight=1e9,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    c_loose = rp.spec.ewridge(
        "c_loose",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        group="g",
        half_life=math.inf,
        ridge=1e-4,
        prior_from="p",
        prior_weight=1e-9,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    c_plain = rp.spec.ewridge(
        "c_plain",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        group="g",
        half_life=math.inf,
        ridge=1e-4,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )

    bank = rp.ModelBank([p_spec, c_tight, c_loose, c_plain])
    out = bank.fit_predict(df)

    # 1. Cold-start row 100: brand-new group "new_grp" predicts immediately from parent's
    # pre-row-100 coefficients (which equals parent's own prediction on row 100)!
    pred_p_100 = float(out["p"].struct.field("pred_y")[100])
    pred_c_100 = float(out["c_tight"].struct.field("pred_y")[100])
    assert math.isfinite(pred_c_100)
    assert pred_c_100 == pytest.approx(pred_p_100, rel=1e-12, abs=1e-12)

    # 2. Extreme limits on new_grp at row 119:
    p_coef = np.asarray(out["p"].struct.field("coef")[119].to_list())
    tight_coef = np.asarray(out["c_tight"].struct.field("coef")[119].to_list())
    loose_coef = np.asarray(out["c_loose"].struct.field("coef")[119].to_list())
    plain_coef = np.asarray(out["c_plain"].struct.field("coef")[119].to_list())
    assert np.allclose(tight_coef, p_coef, rtol=1e-5, atol=1e-5)
    assert np.allclose(loose_coef, plain_coef, rtol=1e-5, atol=1e-5)

    # 3. Bank.predict() on a completely unseen group borrows parent's live coefficients
    # WITHOUT mutating bank state!
    saved_before = bank.save_bytes()
    unseen_df = pl.DataFrame(
        {"t": [125.0], "g": ["never_seen"], "x0": [0.7], "x1": [-1.2], "y": [0.0]}
    )
    pr = bank.predict(unseen_df)
    saved_after = bank.save_bytes()
    assert saved_before == saved_after
    pred_p_unseen = float(pr["p"].struct.field("pred_y")[0])
    pred_c_unseen = float(pr["c_tight"].struct.field("pred_y")[0])
    assert math.isfinite(pred_c_unseen)
    assert pred_c_unseen == pytest.approx(pred_p_unseen, rel=1e-12, abs=1e-12)

    # 4. Warming / null parent fallback: when parent has high min_weight and is not ready,
    # child cleanly uses its own ridge fit without NaNs once child has enough data.
    p_warm = rp.spec.ewridge(
        "pw",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        half_life=math.inf,
        ridge=0.1,
        min_weight=500.0,  # never ready on 120 rows!
        solve_every=0.0,
        max_rows_between_solves=1,
    )
    c_fb = rp.spec.ewridge(
        "cf",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        group="g",
        half_life=math.inf,
        ridge=0.1,
        prior_from="pw",
        prior_weight=5.0,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    c_ref = rp.spec.ewridge(
        "cr",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        group="g",
        half_life=math.inf,
        ridge=0.1,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    out_fb = rp.ModelBank([p_warm, c_fb, c_ref]).fit_predict(df)
    cf_99 = np.asarray(out_fb["cf"].struct.field("coef")[99].to_list())
    cr_99 = np.asarray(out_fb["cr"].struct.field("coef")[99].to_list())
    assert np.all(np.isfinite(cf_99))
    assert np.allclose(cf_99, cr_99, rtol=1e-12, atol=1e-12)


def test_parent_prior_compositions() -> None:
    """3-level DAG, multi-target, sum-scale, window, constraints, QP bounds, and embargo."""
    rng = np.random.default_rng(505)
    n = 90
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    y0 = 0.4 + 0.7 * x0 + 0.3 * x1 + 0.2 * rng.standard_normal(n)
    y1 = -0.3 + 0.2 * x0 + 0.8 * x1 + 0.2 * rng.standard_normal(n)
    t = np.arange(n, dtype=float)
    avail = t + 3.0
    sector = ["tech" if i % 2 == 0 else "fin" for i in range(n)]
    ticker = [f"{sector[i]}_{i % 4}" for i in range(n)]
    df = pl.DataFrame(
        {
            "t": t,
            "avail": avail,
            "sector": sector,
            "ticker": ticker,
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
        }
    )

    # 1. 3-level DAG passed in REVERSE order: [m_ticker, m_sector, m_global]
    g_spec = rp.spec.ewridge(
        "m_global",
        targets=["y0", "y1"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        half_life=30.0,
        ridge=0.05,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    s_spec = rp.spec.ewridge(
        "m_sector",
        targets=["y0", "y1"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        group="sector",
        half_life=30.0,
        ridge=0.05,
        prior_from="m_global",
        prior_weight=5.0,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    t_spec = rp.spec.ewridge(
        "m_ticker",
        targets=["y0", "y1"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        group="ticker",
        half_life=30.0,
        ridge=0.05,
        prior_from="m_sector",
        prior_weight=5.0,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    out_rev = rp.ModelBank([t_spec, s_spec, g_spec]).fit_predict(df)
    out_fwd = rp.ModelBank([g_spec, s_spec, t_spec]).fit_predict(df)
    for name in ("m_global", "m_sector", "m_ticker"):
        assert out_rev[name].equals(out_fwd[name])
    # Multi-target coef column has 2 * 3 = 6 finite entries at the end
    last_tk = out_fwd["m_ticker"].struct.field("coef")[n - 1].to_list()
    assert len(last_tk) == 6 and all(math.isfinite(v) for v in last_tk)

    # Multi-target column coef_prior supports BOTH k_total broadcast (3) and m*k_total (6)
    df_bcast = df.with_columns(pl.lit([0.1, 0.5, 0.5]).alias("p_bcast"))
    out_bcast = rp.ModelBank(
        [
            rp.spec.ewridge(
                "cb",
                targets=["y0", "y1"],
                features=["x0", "x1"],
                clock="t",
                gap_cap=1000.0,
                half_life=30.0,
                ridge=0.05,
                coef_prior="p_bcast",
                prior_weight=4.0,
                solve_every=0.0,
                max_rows_between_solves=1,
                coef_every=0.0,
            )
        ]
    ).fit_predict(df_bcast)
    assert all(math.isfinite(v) for v in out_bcast["cb"].struct.field("coef")[n - 1].to_list())

    # 2. ridge_scale="sum" oracle parity
    p_sum = rp.spec.ewridge(
        "p_sum",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        half_life=30.0,
        ridge=0.2,
        ridge_scale="sum",
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    c_sum = rp.spec.ewridge(
        "c_sum",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        group="sector",
        half_life=30.0,
        ridge=0.2,
        ridge_scale="sum",
        prior_from="p_sum",
        prior_weight=6.0,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    out_sum = rp.ModelBank([p_sum, c_sum]).fit_predict(df)
    mask_tech = np.asarray(sector) == "tech"
    last_tech = int(np.where(mask_tech)[0][-1])
    bp_tech = np.asarray(out_sum["p_sum"].struct.field("coef")[last_tech].to_list())
    got_sum = np.asarray(out_sum["c_sum"].struct.field("coef")[last_tech].to_list())
    t_tech = t[mask_tech]
    row_w = _decay_weights(t_tech, np.ones(len(t_tech)), 30.0)
    ps = 0.5 ** ((t_tech[-1] - t_tech[0]) / 30.0)
    ref_sum, _, _, _ = _scipy_hierarchical_ridge(
        np.column_stack([x0[mask_tech], x1[mask_tech]]),
        y0[mask_tech],
        row_w,
        bp_tech,
        ridge=0.2,
        prior_weight=6.0,
        standardize=False,
        fit_intercept=True,
        prior_include_intercept=True,
        ridge_scale="sum",
        prior_scale=ps,
    )
    assert np.allclose(got_sum, ref_sum, rtol=1e-11, atol=1e-11)

    # 3. Linear equality constraints + QP bounds (simplex) + window_size + embargo/label_clock
    p_base = rp.spec.ewridge(
        "pb",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        half_life=25.0,
        ridge=0.05,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    c_lin = rp.spec.ewridge(
        "c_lin",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        group="sector",
        half_life=25.0,
        ridge=0.05,
        prior_from="pb",
        prior_weight=5.0,
        constraints=([[1.0, 1.0]], [1.0]),
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    c_qp = rp.spec.ewridge(
        "c_qp",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        group="sector",
        half_life=25.0,
        ridge=0.05,
        prior_from="pb",
        prior_weight=5.0,
        coef_min=0.0,
        coef_sum=1.0,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    c_win = rp.spec.ewridge(
        "c_win",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        group="sector",
        half_life=25.0,
        window_size=40.0,
        ridge=0.05,
        prior_from="pb",
        prior_weight=5.0,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_every=0.0,
    )
    out_comp = rp.ModelBank([p_base, c_lin, c_qp, c_win]).fit_predict(df)
    lin_beta = np.asarray(out_comp["c_lin"].struct.field("coef")[last_tech].to_list())
    qp_beta = np.asarray(out_comp["c_qp"].struct.field("coef")[last_tech].to_list())
    win_beta = np.asarray(out_comp["c_win"].struct.field("coef")[last_tech].to_list())
    assert lin_beta[1] + lin_beta[2] == pytest.approx(1.0, abs=1e-12)
    assert qp_beta[1] >= -1e-12 and qp_beta[2] >= -1e-12
    assert qp_beta[1] + qp_beta[2] == pytest.approx(1.0, abs=1e-11)
    assert all(math.isfinite(v) for v in win_beta)

    # Verify QP bounded child against independent SciPy SLSQP oracle
    bp_pb = np.asarray(out_comp["pb"].struct.field("coef")[last_tech].to_list())
    x_t = np.column_stack([x0[mask_tech], x1[mask_tech]])
    y_t = y0[mask_tech]
    rw_t = _decay_weights(t_tech, np.ones(len(t_tech)), 25.0)
    z_t = np.column_stack([np.ones(len(t_tech)), x_t])
    w_t = float(np.sum(rw_t))
    h_qp = (z_t.T * rw_t) @ z_t + np.diag([5.0, w_t * 0.05 + 5.0, w_t * 0.05 + 5.0])
    c_qp_v = z_t.T @ (rw_t * y_t) + 5.0 * bp_pb
    res_qp = minimize(
        lambda v: (0.5 * float(v @ (h_qp @ v)) - float(c_qp_v @ v), h_qp @ v - c_qp_v),
        np.array([0.0, 0.5, 0.5]),
        method="SLSQP",
        jac=True,
        bounds=Bounds([-np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf]),
        constraints=[LinearConstraint(np.array([[0.0, 1.0, 1.0]]), [1.0], [1.0])],
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    assert res_qp.success
    assert np.allclose(qp_beta, res_qp.x, rtol=1e-6, atol=1e-6)

    # 4. Embargo (parent >= child) and label_clock (matching) both accepted and causal
    p_emb = rp.spec.ewridge(
        "p_emb",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        half_life=25.0,
        embargo=5.0,
        ridge=0.05,
    )
    c_emb = rp.spec.ewridge(
        "c_emb",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        group="sector",
        half_life=25.0,
        embargo=3.0,
        ridge=0.05,
        prior_from="p_emb",
        prior_weight=5.0,
    )
    p_lc = rp.spec.ewridge(
        "p_lc",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        label_clock="avail",
        half_life=25.0,
        ridge=0.05,
    )
    c_lc = rp.spec.ewridge(
        "c_lc",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1000.0,
        label_clock="avail",
        group="sector",
        half_life=25.0,
        ridge=0.05,
        prior_from="p_lc",
        prior_weight=5.0,
    )
    out_delay = rp.ModelBank([p_emb, c_emb, p_lc, c_lc]).fit_predict(df)
    assert math.isfinite(float(out_delay["c_emb"].struct.field("pred_y0")[n - 1]))
    assert math.isfinite(float(out_delay["c_lc"].struct.field("pred_y0")[n - 1]))


def test_parent_prior_refusals() -> None:
    """Every invalid spec, DAG, causality, or runtime column shape is refused clearly."""
    base: dict[str, Any] = dict(targets=["y"], features=["x0", "x1"], half_life=20.0)

    # --- Spec-level refusals ---
    with pytest.raises(ValueError, match="prior_weight"):
        rp.spec.ewridge("c", **base, prior_from="p")
    with pytest.raises(ValueError, match="prior_weight"):
        rp.spec.ewridge("c", **base, coef_prior="p_col")
    with pytest.raises(ValueError, match="prior_weight"):
        rp.spec.ewridge("c", **base, prior_weight=5.0)
    for bad_w in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="prior_weight"):
            rp.spec.ewridge("c", **base, prior_from="p", prior_weight=bad_w)
    with pytest.raises(ValueError, match="prior_from"):
        rp.spec.ewridge("c", **base, prior_from="", prior_weight=5.0)
    with pytest.raises(ValueError, match="itself"):
        rp.spec.ewridge("c", **base, prior_from="c", prior_weight=5.0)
    with pytest.raises(ValueError, match="prior_from"):
        rp.spec.ewridge("c", **base, prior_from="p", coef_prior="p_col", prior_weight=5.0)
    with pytest.raises(ValueError, match="prior_include_intercept"):
        rp.spec.ewridge("c", **base, prior_include_intercept=False)
    with pytest.raises(ValueError, match="prior_include_intercept"):
        rp.spec.ewridge(
            "c",
            **base,
            fit_intercept=False,
            prior_from="p",
            prior_weight=5.0,
            prior_include_intercept=False,
        )
    with pytest.raises(ValueError, match="feature_sets"):
        rp.spec.ewridge(
            "c", **base, feature_sets={"a": ["x0"], "b": ["x1"]}, prior_from="p", prior_weight=5.0
        )
    with pytest.raises(ValueError, match="coef_prior"):
        rp.spec.ewridge("c", **base, coef_prior="", prior_weight=5.0)
    with pytest.raises(TypeError, match="coef_prior"):
        rp.spec.ewridge("c", **base, coef_prior=123, prior_weight=5.0)  # type: ignore[arg-type]

    # --- Bank-level DAG & compatibility refusals ---
    with pytest.raises(ValueError, match="names no spec"):
        rp.ModelBank([rp.spec.ewridge("c", **base, prior_from="missing", prior_weight=5.0)])

    # 2-node cycle
    with pytest.raises(ValueError, match="cycle"):
        rp.ModelBank(
            [
                rp.spec.ewridge("a", **base, prior_from="b", prior_weight=5.0),
                rp.spec.ewridge("b", **base, prior_from="a", prior_weight=5.0),
            ]
        )

    # 3-node cycle
    with pytest.raises(ValueError, match="cycle"):
        rp.ModelBank(
            [
                rp.spec.ewridge("a", **base, prior_from="c", prior_weight=5.0),
                rp.spec.ewridge("b", **base, prior_from="a", prior_weight=5.0),
                rp.spec.ewridge("c", **base, prior_from="b", prior_weight=5.0),
            ]
        )

    # Non-linear parent kind
    with pytest.raises(ValueError, match="linear feature coefficients"):
        rp.ModelBank(
            [
                rp.spec.ew_cov("p", features=["x0", "x1"], half_life=20.0),
                rp.spec.ewridge("c", **base, prior_from="p", prior_weight=5.0),
            ]
        )

    # Parent with ridge or half_life grid
    with pytest.raises(ValueError, match="ambiguous"):
        rp.ModelBank(
            [
                rp.spec.ewridge("p", **base, ridge=[0.01, 0.1]),
                rp.spec.ewridge("c", **base, prior_from="p", prior_weight=5.0),
            ]
        )
    with pytest.raises(ValueError, match="ambiguous"):
        rp.ModelBank(
            [
                rp.spec.ewridge("p", targets=["y"], features=["x0", "x1"], half_life=[10.0, 50.0]),
                rp.spec.ewridge("c", **base, prior_from="p", prior_weight=5.0),
            ]
        )

    # Mismatched features (order or names), targets, fit_intercept
    with pytest.raises(ValueError, match="features"):
        rp.ModelBank(
            [
                rp.spec.ewridge("p", targets=["y"], features=["x1", "x0"], half_life=20.0),
                rp.spec.ewridge("c", **base, prior_from="p", prior_weight=5.0),
            ]
        )
    with pytest.raises(ValueError, match="targets"):
        rp.ModelBank(
            [
                rp.spec.ewridge("p", targets=["z"], features=["x0", "x1"], half_life=20.0),
                rp.spec.ewridge("c", **base, prior_from="p", prior_weight=5.0),
            ]
        )
    with pytest.raises(ValueError, match="fit_intercept"):
        rp.ModelBank(
            [
                rp.spec.ewridge("p", **base, fit_intercept=False),
                rp.spec.ewridge("c", **base, fit_intercept=True, prior_from="p", prior_weight=5.0),
            ]
        )

    # Embargo / label_clock causality refusals
    with pytest.raises(ValueError, match="embargo"):
        rp.ModelBank(
            [
                rp.spec.ewridge("p", **base, clock="t", gap_cap=1000.0),
                rp.spec.ewridge(
                    "c",
                    **base,
                    clock="t",
                    gap_cap=1000.0,
                    embargo=5.0,
                    prior_from="p",
                    prior_weight=5.0,
                ),
            ]
        )
    with pytest.raises(ValueError, match="embargo"):
        rp.ModelBank(
            [
                rp.spec.ewridge("p", **base, clock="t", gap_cap=1000.0, embargo=2.0),
                rp.spec.ewridge(
                    "c",
                    **base,
                    clock="t",
                    gap_cap=1000.0,
                    embargo=5.0,
                    prior_from="p",
                    prior_weight=5.0,
                ),
            ]
        )
    with pytest.raises(ValueError, match="label_clock"):
        rp.ModelBank(
            [
                rp.spec.ewridge("p", **base, clock="t", gap_cap=1000.0),
                rp.spec.ewridge(
                    "c",
                    **base,
                    clock="t",
                    gap_cap=1000.0,
                    label_clock="avail",
                    prior_from="p",
                    prior_weight=5.0,
                ),
            ]
        )
    with pytest.raises(ValueError, match="label_clock"):
        rp.ModelBank(
            [
                rp.spec.ewridge("p", **base, clock="t", gap_cap=1000.0, label_clock="avail_a"),
                rp.spec.ewridge(
                    "c",
                    **base,
                    clock="t",
                    gap_cap=1000.0,
                    label_clock="avail_b",
                    prior_from="p",
                    prior_weight=5.0,
                ),
            ]
        )

    # --- Runtime column coef_prior dtype & length refusals ---
    df_bad_dtype = pl.DataFrame(
        {"x0": [1.0, 2.0], "x1": [0.5, 1.5], "y": [1.0, 2.0], "bad": [1.0, 2.0]}
    )
    with pytest.raises((ValueError, TypeError), match="List|coef_prior|bad"):
        rp.ModelBank(
            [rp.spec.ewridge("c", **base, coef_prior="bad", prior_weight=5.0)]
        ).fit_predict(df_bad_dtype)

    df_bad_len = pl.DataFrame(
        {"x0": [1.0, 2.0], "x1": [0.5, 1.5], "y": [1.0, 2.0], "bad": [[1.0, 2.0], [1.0, 2.0]]}
    )
    with pytest.raises(ValueError, match="length"):
        rp.ModelBank(
            [rp.spec.ewridge("c", **base, coef_prior="bad", prior_weight=5.0)]
        ).fit_predict(df_bad_len)
