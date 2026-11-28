"""Regression variants: unit tests, invariants, refusals, and 50-seed oracle grids.

Covers:
1. ``ewridge(evidence=True, evidence_prior=...)`` vs ``sklearn.linear_model.BayesianRidge``
2. ``sgd(average=True)`` and ``pa(average=True)`` vs ``sklearn.linear_model.SGDRegressor``
3. ``lasso(positive=True)`` vs ``sklearn.linear_model.Lasso`` / ``ElasticNet``
4. ``huber(huber_norm=...)`` (tukey, andrews, hampel, ramsay) vs ``statsmodels.robust.RLM``
5. ``sgd(l1=..., schedule="optimal", invariant=True)`` vs ``SGDRegressor`` and
   Karampatziakis-Langford's exact importance-weight-invariant step.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, SGDRegressor

import ledger
from rheon import polars as rp

TIER = "mixed"

SEEDS = tuple(range(50))


# --- helpers -----------------------------------------------------------------


def _coef_vec(bank: rp.ModelBank, spec_name: str, *, penalty: float | None = None) -> np.ndarray:
    df = bank.coef().filter(pl.col("spec") == spec_name)
    if penalty is not None:
        df = df.filter(pl.col("penalty") == penalty)
    return df["coef"].to_numpy()


def _linear_frame(seed: int, n: int, k: int, *, noise: float = 0.25, bias: float = 0.6) -> tuple:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k))
    beta = rng.uniform(-1.5, 1.5, size=k)
    y = bias + x @ beta + noise * rng.standard_normal(n)
    cols = {f"x{i}": x[:, i] for i in range(k)}
    cols["y"] = y
    return pl.DataFrame(cols), x, y, beta, bias


# --- oracle-ledger runners (Rheon-only outputs for ledger.guard) -------------


def _ewridge_evidence_outputs() -> dict[str, np.ndarray]:
    k = 4
    feats = [f"x{i}" for i in range(k)]
    n = 120
    c_def, s_def, c_ml2, s_ml2, c_dec, s_dec = [], [], [], [], [], []
    for seed in SEEDS:
        df, _, _, _, _ = _linear_frame(1000 + seed, n, k, noise=0.3)
        specs = [
            rp.spec.ewridge(
                "ev_def",
                targets=["y"],
                features=feats,
                half_life=math.inf,
                evidence=True,
                solve_every=0.0,
                max_rows_between_solves=1,
                min_weight=0.0,
                max_error_inflation=math.inf,
            ),
            rp.spec.ewridge(
                "ev_ml2",
                targets=["y"],
                features=feats,
                half_life=math.inf,
                evidence=True,
                evidence_prior=0.0,
                solve_every=0.0,
                max_rows_between_solves=1,
                min_weight=0.0,
                max_error_inflation=math.inf,
            ),
            rp.spec.ewridge(
                "ev_dec",
                targets=["y"],
                features=feats,
                half_life=40.0,
                evidence=True,
                solve_every=0.0,
                max_rows_between_solves=1,
                min_weight=0.0,
                max_error_inflation=math.inf,
            ),
        ]
        bank = rp.ModelBank(specs)
        bank.fit(df.slice(0, n - 1))
        pred = bank.predict(df.slice(n - 1, 1))
        c_def.append(_coef_vec(bank, "ev_def"))
        c_ml2.append(_coef_vec(bank, "ev_ml2"))
        c_dec.append(_coef_vec(bank, "ev_dec"))
        s_def.append(pred["ev_def"].struct.field("pred_std_y")[0])
        s_ml2.append(pred["ev_ml2"].struct.field("pred_std_y")[0])
        s_dec.append(pred["ev_dec"].struct.field("pred_std_y")[0])
    return {
        "c_def": np.asarray(c_def),
        "s_def": np.asarray(s_def),
        "c_ml2": np.asarray(c_ml2),
        "s_ml2": np.asarray(s_ml2),
        "c_dec": np.asarray(c_dec),
        "s_dec": np.asarray(s_dec),
    }


def _averaged_sgd_pa_outputs() -> dict[str, np.ndarray]:
    k = 4
    feats = [f"x{i}" for i in range(k)]
    n = 150
    c_sgd, c_pa1, c_pa2 = [], [], []
    for seed in SEEDS:
        df, _, _, _, _ = _linear_frame(2000 + seed, n, k, noise=0.2, bias=0.0)
        specs = [
            rp.spec.sgd(
                "sgd_avg",
                targets=["y"],
                features=feats,
                fit_intercept=False,
                half_life=math.inf,
                learning_rate=0.02,
                l2=0.01,
                standardize=False,
                clip_gradient=math.inf,
                average=True,
                min_weight=0.0,
            ),
            rp.spec.pa(
                "pa1_avg",
                targets=["y"],
                features=feats,
                fit_intercept=False,
                half_life=math.inf,
                mode="pa1",
                c=0.5,
                eps=0.0,
                standardize=False,
                average=True,
                min_weight=0.0,
            ),
            rp.spec.pa(
                "pa2_avg",
                targets=["y"],
                features=feats,
                fit_intercept=False,
                half_life=math.inf,
                mode="pa2",
                c=0.5,
                eps=0.0,
                standardize=False,
                average=True,
                min_weight=0.0,
            ),
        ]
        bank = rp.ModelBank(specs)
        bank.fit(df)
        c_sgd.append(_coef_vec(bank, "sgd_avg"))
        c_pa1.append(_coef_vec(bank, "pa1_avg"))
        c_pa2.append(_coef_vec(bank, "pa2_avg"))
    return {
        "c_sgd": np.asarray(c_sgd),
        "c_pa1": np.asarray(c_pa1),
        "c_pa2": np.asarray(c_pa2),
    }


def _lasso_positive_outputs() -> dict[str, np.ndarray]:
    n_small, k_small = 140, 6
    n_split, k_split = 180, 44
    f_small = [f"x{i}" for i in range(k_small)]
    f_split = [f"x{i}" for i in range(k_split)]
    c_small, c_enet, c_split = [], [], []
    for seed in SEEDS:
        df_s, _, _, _, _ = _linear_frame(3000 + seed, n_small, k_small, noise=0.2)
        bank_s = rp.ModelBank(
            [
                rp.spec.lasso(
                    "las_s",
                    targets=["y"],
                    features=f_small,
                    half_life=math.inf,
                    penalties=[0.1],
                    l1_ratio=1.0,
                    positive=True,
                    solve_every=0.0,
                    max_rows_between_solves=1,
                    max_iter=1000,
                    tol=1e-13,
                    min_weight=0.0,
                ),
                rp.spec.lasso(
                    "enet_s",
                    targets=["y"],
                    features=f_small,
                    half_life=math.inf,
                    penalties=[0.1],
                    l1_ratio=0.6,
                    positive=True,
                    solve_every=0.0,
                    max_rows_between_solves=1,
                    max_iter=1000,
                    tol=1e-13,
                    min_weight=0.0,
                ),
            ]
        )
        bank_s.fit(df_s)
        c_small.append(_coef_vec(bank_s, "las_s"))
        c_enet.append(_coef_vec(bank_s, "enet_s"))

        df_w, _, _, _, _ = _linear_frame(3500 + seed, n_split, k_split, noise=0.2)
        bank_w = rp.ModelBank(
            [
                rp.spec.lasso(
                    "las_w",
                    targets=["y"],
                    features=f_split,
                    half_life=math.inf,
                    penalties=[0.15],
                    l1_ratio=1.0,
                    positive=True,
                    solve_every=0.0,
                    max_rows_between_solves=1,
                    max_iter=2000,
                    tol=1e-13,
                    min_weight=0.0,
                )
            ]
        )
        bank_w.fit(df_w)
        c_split.append(_coef_vec(bank_w, "las_w"))
    return {
        "c_small": np.asarray(c_small),
        "c_enet": np.asarray(c_enet),
        "c_split": np.asarray(c_split),
    }


def _huber_redescending_frame(seed: int, n: int = 201, k: int = 3) -> tuple:
    rng = np.random.default_rng(4000 + seed)
    x = rng.standard_normal((n, k))
    beta = np.array([1.5, -1.2, 0.8])
    bias = 0.7
    y = bias + x @ beta + 0.25 * rng.standard_normal(n)
    outliers = rng.random(n) < 0.08
    y[outliers] += rng.choice([-1.0, 1.0], size=int(outliers.sum())) * 20.0
    cols = {f"x{i}": x[:, i] for i in range(k)}
    cols["y"] = y
    return pl.DataFrame(cols), x, y


def _huber_redescending_outputs() -> dict[str, np.ndarray]:
    norms = ("tukey", "andrews", "hampel", "ramsay")
    feats = ["x0", "x1", "x2"]
    n = 201
    out: dict[str, list[np.ndarray]] = {nm: [] for nm in norms}
    for seed in SEEDS:
        df, _, _ = _huber_redescending_frame(seed, n=n)
        specs = [
            rp.spec.huber(
                nm,
                targets=["y"],
                features=feats,
                half_life=math.inf,
                huber_norm=nm,
                huber_scale="mad",
                ridge=1e-12,
                huber_refit_rows=n,
                solve_every=0.0,
                max_rows_between_solves=1,
                min_weight=0.0,
            )
            for nm in norms
        ]
        bank = rp.ModelBank(specs)
        bank.fit(df)
        for nm in norms:
            out[nm].append(_coef_vec(bank, nm))
    return {nm: np.asarray(v) for nm, v in out.items()}


def _sgd_variants_frame(seed: int, n: int = 150, k: int = 4) -> tuple:
    rng = np.random.default_rng(5000 + seed)
    # Scale features to ||x|| ~ 0.5 so Bottou's optimal schedule at alpha=0.02
    # stays strictly within the squared-loss stability bound on both sides.
    x = 0.25 * rng.standard_normal((n, k))
    beta = rng.uniform(-1.5, 1.5, size=k)
    y = x @ beta + 0.1 * rng.standard_normal(n)
    w = np.random.default_rng(6000 + seed).uniform(0.2, 5.0, size=n)
    cols = {f"x{i}": x[:, i] for i in range(k)}
    cols["y"] = y
    cols["w"] = w
    return pl.DataFrame(cols), x, y, w


def _sgd_variants_outputs() -> dict[str, np.ndarray]:
    k = 4
    feats = [f"x{i}" for i in range(k)]
    n = 150
    c_l1, c_elopt, c_optavg, c_inv = [], [], [], []
    for seed in SEEDS:
        df_w, _, _, _ = _sgd_variants_frame(seed, n=n, k=k)
        specs = [
            rp.spec.sgd(
                "sgd_l1",
                targets=["y"],
                features=feats,
                fit_intercept=False,
                half_life=math.inf,
                learning_rate=0.02,
                l1=0.01,
                l2=0.0,
                standardize=False,
                clip_gradient=math.inf,
                min_weight=0.0,
            ),
            rp.spec.sgd(
                "sgd_elopt",
                targets=["y"],
                features=feats,
                fit_intercept=False,
                half_life=math.inf,
                schedule="optimal",
                l1=0.005,
                l2=0.015,
                standardize=False,
                clip_gradient=math.inf,
                min_weight=0.0,
            ),
            rp.spec.sgd(
                "sgd_optavg",
                targets=["y"],
                features=feats,
                fit_intercept=False,
                half_life=math.inf,
                schedule="optimal",
                l2=0.02,
                average=True,
                standardize=False,
                clip_gradient=math.inf,
                min_weight=0.0,
            ),
            rp.spec.sgd(
                "sgd_inv",
                targets=["y"],
                features=feats,
                fit_intercept=False,
                half_life=math.inf,
                weight="w",
                learning_rate=0.25,
                invariant=True,
                standardize=False,
                clip_gradient=math.inf,
                min_weight=0.0,
            ),
        ]
        bank = rp.ModelBank(specs)
        bank.fit(df_w)
        c_l1.append(_coef_vec(bank, "sgd_l1"))
        c_elopt.append(_coef_vec(bank, "sgd_elopt"))
        c_optavg.append(_coef_vec(bank, "sgd_optavg"))
        c_inv.append(_coef_vec(bank, "sgd_inv"))
    return {
        "c_l1": np.asarray(c_l1),
        "c_elopt": np.asarray(c_elopt),
        "c_optavg": np.asarray(c_optavg),
        "c_inv": np.asarray(c_inv),
    }


ledger.guard("ewridge_evidence", _ewridge_evidence_outputs, oracles=["scikit-learn"])
ledger.guard("averaged_sgd_pa", _averaged_sgd_pa_outputs, oracles=["scikit-learn"])
ledger.guard("lasso_positive", _lasso_positive_outputs, oracles=["scikit-learn"])
ledger.guard("huber_redescending", _huber_redescending_outputs, oracles=["statsmodels"])
ledger.guard("sgd_variants", _sgd_variants_outputs, oracles=["scikit-learn"])


# --- 50-seed oracle-ledger grids ---------------------------------------------


@pytest.mark.oracle_grid(reason="50-seed BayesianRidge evidence oracle grid")
def test_ewridge_evidence_matches_sklearn_bayesian_ridge_50_seeds():
    got = _ewridge_evidence_outputs()
    n, k = 120, 4
    def_c, def_s, ml2_c, ml2_s, dec_c, dec_s = [], [], [], [], [], []
    # Exponential weights for half_life=40 on n-1 rows, scaled to Kish n.
    ages = np.arange(n - 2, -1, -1, dtype=float)
    raw_w = 0.5 ** (ages / 40.0)
    n_kish = (raw_w.sum() ** 2) / np.square(raw_w).sum()
    sw_dec = raw_w * (n_kish / raw_w.sum())

    for idx, seed in enumerate(SEEDS):
        _, x, y, _, _ = _linear_frame(1000 + seed, n, k, noise=0.3)
        x_tr, y_tr, x_te = x[:-1], y[:-1], x[-1:]

        br_def = BayesianRidge(tol=1e-14, max_iter=5000).fit(x_tr, y_tr)
        _, std_def = br_def.predict(x_te, return_std=True)
        want_def = np.concatenate([[br_def.intercept_], br_def.coef_])
        def_c.append(float(np.max(np.abs(got["c_def"][idx] - want_def))))
        def_s.append(float(abs(got["s_def"][idx] - std_def[0])))

        br_ml2 = BayesianRidge(
            alpha_1=0.0, alpha_2=0.0, lambda_1=0.0, lambda_2=0.0, tol=1e-14, max_iter=5000
        ).fit(x_tr, y_tr)
        _, std_ml2 = br_ml2.predict(x_te, return_std=True)
        want_ml2 = np.concatenate([[br_ml2.intercept_], br_ml2.coef_])
        ml2_c.append(float(np.max(np.abs(got["c_ml2"][idx] - want_ml2))))
        ml2_s.append(float(abs(got["s_ml2"][idx] - std_ml2[0])))

        br_dec = BayesianRidge(tol=1e-14, max_iter=5000).fit(x_tr, y_tr, sample_weight=sw_dec)
        _, std_dec = br_dec.predict(x_te, return_std=True)
        want_dec = np.concatenate([[br_dec.intercept_], br_dec.coef_])
        dec_c.append(float(np.max(np.abs(got["c_dec"][idx] - want_dec))))
        dec_s.append(float(abs(got["s_dec"][idx] - std_dec[0])))

    w, _ = ledger.record("ewridge_evidence", "default_coef_err", def_c, floor=1e-11)
    assert w < 1e-9
    w, _ = ledger.record("ewridge_evidence", "default_pred_std_err", def_s, floor=1e-11)
    assert w < 1e-9
    w, _ = ledger.record("ewridge_evidence", "ml2_coef_err", ml2_c, floor=1e-11)
    assert w < 1e-9
    w, _ = ledger.record("ewridge_evidence", "ml2_pred_std_err", ml2_s, floor=1e-11)
    assert w < 1e-9
    w, _ = ledger.record("ewridge_evidence", "decayed_coef_err", dec_c, floor=1e-11)
    assert w < 1e-9
    w, _ = ledger.record("ewridge_evidence", "decayed_pred_std_err", dec_s, floor=1e-11)
    assert w < 1e-9


@pytest.mark.oracle_grid(reason="50-seed averaged SGD and PA oracle grid")
def test_averaged_sgd_and_pa_match_sklearn_50_seeds():
    got = _averaged_sgd_pa_outputs()
    n, k = 150, 4
    sgd_e, pa1_e, pa2_e = [], [], []
    for idx, seed in enumerate(SEEDS):
        _, x, y, _, _ = _linear_frame(2000 + seed, n, k, noise=0.2, bias=0.0)

        sk_sgd = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=0.01,
            learning_rate="constant",
            eta0=0.02,
            fit_intercept=False,
            average=True,
            max_iter=1,
            tol=None,
            shuffle=False,
        ).fit(x, y)
        sgd_e.append(float(np.max(np.abs(got["c_sgd"][idx] - sk_sgd.coef_))))

        sk_pa1 = SGDRegressor(
            loss="epsilon_insensitive",
            epsilon=0.0,
            penalty=None,
            learning_rate="pa1",
            eta0=0.5,
            fit_intercept=False,
            average=True,
            max_iter=1,
            tol=None,
            shuffle=False,
        ).fit(x, y)
        pa1_e.append(float(np.max(np.abs(got["c_pa1"][idx] - sk_pa1.coef_))))

        sk_pa2 = SGDRegressor(
            loss="epsilon_insensitive",
            epsilon=0.0,
            penalty=None,
            learning_rate="pa2",
            eta0=0.5,
            fit_intercept=False,
            average=True,
            max_iter=1,
            tol=None,
            shuffle=False,
        ).fit(x, y)
        pa2_e.append(float(np.max(np.abs(got["c_pa2"][idx] - sk_pa2.coef_))))

    w, _ = ledger.record("averaged_sgd_pa", "sgd_avg_coef_err", sgd_e, floor=1e-13)
    assert w < 1e-11
    w, _ = ledger.record("averaged_sgd_pa", "pa1_avg_coef_err", pa1_e, floor=1e-13)
    assert w < 1e-11
    w, _ = ledger.record("averaged_sgd_pa", "pa2_avg_coef_err", pa2_e, floor=1e-13)
    assert w < 1e-11


@pytest.mark.oracle_grid(reason="50-seed positive Lasso and ElasticNet oracle grid")
def test_lasso_positive_matches_sklearn_50_seeds():
    got = _lasso_positive_outputs()
    n_small, k_small = 140, 6
    n_split, k_split = 180, 44
    small_e, enet_e, split_e = [], [], []
    for idx, seed in enumerate(SEEDS):
        _, xs, ys, _, _ = _linear_frame(3000 + seed, n_small, k_small, noise=0.2)
        sd_s = xs.std(axis=0, ddof=0)
        xs_std = xs / sd_s
        sk_l = Lasso(alpha=0.1, positive=True, fit_intercept=True, tol=1e-14, max_iter=20000).fit(
            xs_std, ys
        )
        want_l = np.concatenate([[sk_l.intercept_], sk_l.coef_ / sd_s])
        small_e.append(float(np.max(np.abs(got["c_small"][idx] - want_l))))

        sk_e = ElasticNet(
            alpha=0.1, l1_ratio=0.6, positive=True, fit_intercept=True, tol=1e-14, max_iter=20000
        ).fit(xs_std, ys)
        want_e = np.concatenate([[sk_e.intercept_], sk_e.coef_ / sd_s])
        enet_e.append(float(np.max(np.abs(got["c_enet"][idx] - want_e))))

        _, xw, yw, _, _ = _linear_frame(3500 + seed, n_split, k_split, noise=0.2)
        sd_w = xw.std(axis=0, ddof=0)
        xw_std = xw / sd_w
        sk_w = Lasso(alpha=0.15, positive=True, fit_intercept=True, tol=1e-14, max_iter=20000).fit(
            xw_std, yw
        )
        want_w = np.concatenate([[sk_w.intercept_], sk_w.coef_ / sd_w])
        split_e.append(float(np.max(np.abs(got["c_split"][idx] - want_w))))

    w, _ = ledger.record("lasso_positive", "lasso_small_coef_err", small_e, floor=1e-10)
    assert w < 1e-7
    w, _ = ledger.record("lasso_positive", "enet_pos_coef_err", enet_e, floor=1e-10)
    assert w < 1e-7
    w, _ = ledger.record("lasso_positive", "lasso_split_coef_err", split_e, floor=1e-10)
    assert w < 1e-7


@pytest.mark.oracle_grid(reason="50-seed redescending Huber RLM oracle grid")
def test_huber_redescending_matches_statsmodels_rlm_50_seeds():
    got = _huber_redescending_outputs()
    sm_norms = {
        "tukey": sm.robust.norms.TukeyBiweight(c=4.685),
        "andrews": sm.robust.norms.AndrewWave(a=1.339),
        "hampel": sm.robust.norms.Hampel(a=2.0, b=4.0, c=8.0),
        "ramsay": sm.robust.norms.RamsayE(a=0.3),
    }
    errs: dict[str, list[float]] = {nm: [] for nm in sm_norms}
    for idx, seed in enumerate(SEEDS):
        _, x, y = _huber_redescending_frame(seed, n=201)
        exog = sm.add_constant(x)
        for nm, m_norm in sm_norms.items():
            # Huber warm-start matches our refit's initialization so both
            # converge to the same local mode of the redescending objective.
            hub_start = sm.RLM(y, exog, M=sm.robust.norms.HuberT(t=1.345)).fit(
                maxiter=200, tol=1e-12
            )
            res = sm.RLM(y, exog, M=m_norm).fit(
                start_params=hub_start.params, maxiter=200, tol=1e-12
            )
            errs[nm].append(float(np.max(np.abs(got[nm][idx] - res.params))))

    for nm in ("tukey", "andrews", "hampel", "ramsay"):
        w, _ = ledger.record("huber_redescending", f"{nm}_coef_err", errs[nm], floor=1e-7)
        assert w < 1e-4, f"{nm}: worst coef error vs statsmodels RLM was {w}"


@pytest.mark.oracle_grid(reason="50-seed SGD l1, optimal schedule, and invariant oracle grid")
def test_sgd_elastic_optimal_invariant_50_seeds():
    got = _sgd_variants_outputs()
    n, k = 150, 4
    l1_e, elopt_e, optavg_e, inv_e = [], [], [], []
    for idx, seed in enumerate(SEEDS):
        _, x, y, w = _sgd_variants_frame(seed, n=n, k=k)

        sk_l1 = SGDRegressor(
            loss="squared_error",
            penalty="l1",
            alpha=0.01,
            learning_rate="constant",
            eta0=0.02,
            fit_intercept=False,
            max_iter=1,
            tol=None,
            shuffle=False,
        ).fit(x, y)
        l1_e.append(float(np.max(np.abs(got["c_l1"][idx] - sk_l1.coef_))))

        sk_elopt = SGDRegressor(
            loss="squared_error",
            penalty="elasticnet",
            alpha=0.02,
            l1_ratio=0.25,
            learning_rate="optimal",
            fit_intercept=False,
            max_iter=1,
            tol=None,
            shuffle=False,
        ).fit(x, y)
        elopt_e.append(float(np.max(np.abs(got["c_elopt"][idx] - sk_elopt.coef_))))

        sk_optavg = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=0.02,
            learning_rate="optimal",
            average=True,
            fit_intercept=False,
            max_iter=1,
            tol=None,
            shuffle=False,
        ).fit(x, y)
        optavg_e.append(float(np.max(np.abs(got["c_optavg"][idx] - sk_optavg.coef_))))

        # Exact Karampatziakis-Langford invariant update recursion
        beta = np.zeros(k)
        for i in range(n):
            xi = x[i]
            zz = float(xi @ xi)
            s = -math.expm1(-0.25 * w[i] * zz) / zz if zz > 0.0 else 0.0
            err = float(xi @ beta) - y[i]
            beta = beta - s * err * xi
        inv_e.append(float(np.max(np.abs(got["c_inv"][idx] - beta))))

    w, _ = ledger.record("sgd_variants", "l1_coef_err", l1_e, floor=1e-13)
    assert w < 1e-11
    w, _ = ledger.record("sgd_variants", "elastic_optimal_coef_err", elopt_e, floor=1e-12)
    assert w < 1e-10
    w, _ = ledger.record("sgd_variants", "optimal_avg_coef_err", optavg_e, floor=1e-12)
    assert w < 1e-10
    w, _ = ledger.record("sgd_variants", "invariant_exact_err", inv_e, floor=1e-13)
    assert w < 1e-11


# --- fast essential tests (invariants, edge cases, defaults, refusals) -------


def test_defaults_are_bit_identical_and_specs_roundtrip():
    df, _, _, _, _ = _linear_frame(77, 80, 3)
    pairs = [
        (
            rp.spec.ewridge("a", targets=["y"], features=["x0", "x1", "x2"], half_life=50.0),
            rp.spec.ewridge(
                "b", targets=["y"], features=["x0", "x1", "x2"], half_life=50.0, evidence=False
            ),
        ),
        (
            rp.spec.lasso(
                "a", targets=["y"], features=["x0", "x1", "x2"], half_life=50.0, penalties=[0.1]
            ),
            rp.spec.lasso(
                "b",
                targets=["y"],
                features=["x0", "x1", "x2"],
                half_life=50.0,
                penalties=[0.1],
                positive=False,
            ),
        ),
        (
            rp.spec.huber("a", targets=["y"], features=["x0", "x1", "x2"], half_life=50.0),
            rp.spec.huber(
                "b",
                targets=["y"],
                features=["x0", "x1", "x2"],
                half_life=50.0,
                huber_norm="huber",
            ),
        ),
        (
            rp.spec.pa("a", targets=["y"], features=["x0", "x1", "x2"], half_life=50.0),
            rp.spec.pa(
                "b", targets=["y"], features=["x0", "x1", "x2"], half_life=50.0, average=False
            ),
        ),
        (
            rp.spec.sgd("a", targets=["y"], features=["x0", "x1", "x2"], half_life=50.0),
            rp.spec.sgd(
                "b",
                targets=["y"],
                features=["x0", "x1", "x2"],
                half_life=50.0,
                average=False,
                invariant=False,
            ),
        ),
    ]
    for s_a, s_b in pairs:
        bank_a = rp.ModelBank([s_a])
        bank_b = rp.ModelBank([s_b])
        assert bank_a.specs[0] == s_a
        assert bank_b.specs[0] == s_b
        out_a = bank_a.fit_predict(df)["a"].struct.field("pred_y").to_numpy()
        out_b = bank_b.fit_predict(df)["b"].struct.field("pred_y").to_numpy()
        np.testing.assert_array_equal(out_a, out_b)


def test_chunk_invariance_save_load_and_zero_weight_rule9():
    df, _, _, _, _ = _linear_frame(88, 120, 3)
    rng = np.random.default_rng(89)
    w = rng.uniform(0.5, 2.0, size=120)
    t = np.arange(120, dtype=float)
    df = df.with_columns(pl.Series("t", t), pl.Series("w", w))

    specs = [
        rp.spec.ewridge(
            "ev",
            targets=["y"],
            features=["x0", "x1", "x2"],
            clock="t",
            gap_cap=10.0,
            half_life=40.0,
            weight="w",
            evidence=True,
        ),
        rp.spec.lasso(
            "lpos",
            targets=["y"],
            features=["x0", "x1", "x2"],
            clock="t",
            gap_cap=10.0,
            half_life=40.0,
            weight="w",
            penalties=[0.2, 0.05],
            positive=True,
        ),
        rp.spec.huber(
            "tukey",
            targets=["y"],
            features=["x0", "x1", "x2"],
            clock="t",
            gap_cap=10.0,
            half_life=40.0,
            weight="w",
            huber_norm="tukey",
            huber_refit_rows=40,
            emit_huber_scale=True,
        ),
        rp.spec.pa(
            "pa_avg",
            targets=["y"],
            features=["x0", "x1", "x2"],
            clock="t",
            gap_cap=10.0,
            half_life=40.0,
            weight="w",
            average=True,
        ),
        rp.spec.sgd(
            "sgd_var",
            targets=["y"],
            features=["x0", "x1", "x2"],
            clock="t",
            gap_cap=10.0,
            half_life=40.0,
            weight="w",
            schedule="optimal",
            l1=0.01,
            l2=0.05,
            average=True,
            invariant=True,
        ),
    ]

    # Single chunk vs 12 chunks + save/load round-trip at row 60
    one = rp.ModelBank(specs).fit_predict(df)

    b_chunked = rp.ModelBank(specs)
    parts = [b_chunked.fit_predict(df.slice(i, 10)) for i in range(0, 120, 10)]
    many = pl.concat(parts)

    b_save = rp.ModelBank(specs)
    p1 = b_save.fit_predict(df.slice(0, 60))
    blob = b_save.save_bytes()
    b_load = rp.ModelBank.load_bytes(blob, specs)
    p2 = b_load.fit_predict(df.slice(60, 60))
    reloaded = pl.concat([p1, p2])

    for s in specs:
        name = s["name"]
        assert b_load.specs == specs
        pred_fields = [f for f in one[name].struct.fields if f.startswith("pred_")]
        assert pred_fields
        for field in pred_fields:
            a = one[name].struct.field(field).to_numpy()
            b = many[name].struct.field(field).to_numpy()
            c = reloaded[name].struct.field(field).to_numpy()
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a, c)

    # Zero-weight rows (Rule 9): insert a weight=0 row at t=50.5 with extreme y;
    # predictions on all original rows must be bit-identical!
    zrow = pl.DataFrame(
        {"x0": [50.0], "x1": [-50.0], "x2": [10.0], "y": [1e6], "t": [50.5], "w": [0.0]}
    )
    df_with_zero = pl.concat([df.slice(0, 51), zrow, df.slice(51, 69)])
    out_z = rp.ModelBank(specs).fit_predict(df_with_zero).filter(pl.col("w") > 0)
    for s in specs:
        name = s["name"]
        for field in [f for f in one[name].struct.fields if f.startswith("pred_")]:
            np.testing.assert_array_equal(
                one[name].struct.field(field).to_numpy(),
                out_z[name].struct.field(field).to_numpy(),
            )

    # Out-of-sample (Rule 2): perturbing row 80's target must leave row 80's
    # prediction and pred_std bit-identical!
    df_pert = df.with_columns(
        pl.when(pl.arange(0, 120) == 80).then(pl.col("y") + 1e4).otherwise(pl.col("y")).alias("y")
    )
    out_pert = rp.ModelBank(specs).fit_predict(df_pert)
    for s in specs:
        name = s["name"]
        for field in [f for f in one[name].struct.fields if f.startswith("pred_")]:
            a = one[name].struct.field(field)[80]
            b = out_pert[name].struct.field(field)[80]
            assert (a is None and b is None) or a == b


def test_ewridge_evidence_features_feature_sets_and_withholding():
    df, _, _, _, _ = _linear_frame(99, 100, 3)
    spec = rp.spec.ewridge(
        "ev",
        targets=["y"],
        features=["x0", "x1", "x2"],
        feature_sets={"a": ["x0", "x1"], "all": ["x0", "x1", "x2"]},
        half_life=50.0,
        evidence=True,
        evidence_prior=[1e-5, 1e-5, 1e-6, 1e-6],
        standardize=True,
        min_weight=15.0,
        emit_se_coef=True,
        emit_error_inflation=True,
    )
    idx = rp.spec.output_index(spec)
    assert "pred_std_y__a" in idx["field"].to_list()
    assert "pred_std_y__all" in idx["field"].to_list()
    assert idx.filter(pl.col("kind") == "pred_std")["ridge"].is_null().all()

    out = rp.ModelBank([spec]).fit_predict(df)
    ev = out["ev"]
    # Wherever pred is null (withheld), pred_std is null too; wherever pred is finite, pred_std > 0.
    for suf in ("__a", "__all"):
        p = ev.struct.field(f"pred_y{suf}").is_null()
        s = ev.struct.field(f"pred_std_y{suf}")
        assert (p == s.is_null()).all()
        valid = s.drop_nulls()
        assert valid.len() > 50
        assert (valid > 0).all()


def test_huber_norms_match_statsmodels_weights_and_redescending_keeps_scale():
    # Verify extreme outlier rows (w_rob = 0 under Tukey/Andrews/Hampel) still update huber_scale.
    df, _, _ = _huber_redescending_frame(12, n=120)
    for nm in ("tukey", "andrews", "hampel", "ramsay"):
        spec = rp.spec.huber(
            "h",
            targets=["y"],
            features=["x0", "x1", "x2"],
            half_life=50.0,
            huber_norm=nm,
            emit_huber_scale=True,
            huber_refit_rows=30,
        )
        out = rp.ModelBank([spec]).fit_predict(df)
        scales = out["h"].struct.field("huber_scale_y").drop_nulls()
        assert scales.len() > 80
        assert (scales > 0).all()


def test_sgd_invariant_never_overshoots_under_huge_weights():
    df = pl.DataFrame({"x0": [2.0] * 10, "y": [4.0] * 10, "w": [1e6] * 10})
    spec = rp.spec.sgd(
        "inv",
        targets=["y"],
        features=["x0"],
        fit_intercept=False,
        half_life=math.inf,
        weight="w",
        learning_rate=0.5,
        standardize=False,
        invariant=True,
        min_weight=0.0,
    )
    out = rp.ModelBank([spec]).fit_predict(df)
    preds = out["inv"].struct.field("pred_y").to_numpy()
    assert np.all(np.isfinite(preds))
    # First row predicts 0, update contracts exact residual to 4.0, so row 1..9 predict 4.0!
    assert preds[0] == 0.0
    assert np.allclose(preds[1:], 4.0, atol=1e-12)


def test_validation_refusals():
    base = dict(targets=["y"], features=["x0", "x1"], half_life=50.0)

    # ewridge evidence refusals
    with pytest.raises(ValueError, match="evidence=True chooses the ridge"):
        rp.spec.ewridge("m", evidence=True, ridge=0.1, **base)
    with pytest.raises(ValueError, match='ridge_scale="sum"'):
        rp.spec.ewridge("m", evidence=True, ridge_scale="sum", **base)
    with pytest.raises(ValueError, match="coef_prior"):
        rp.spec.ewridge("m", evidence=True, coef_prior=[[0.0, 0.0, 0.0]], **base)
    with pytest.raises(ValueError, match="evidence_prior applies only under evidence=True"):
        rp.spec.ewridge("m", evidence=False, evidence_prior=1e-6, **base)
    with pytest.raises(ValueError, match="single number or a list of 4"):
        rp.spec.ewridge("m", evidence=True, evidence_prior=[1e-6, 1e-6], **base)
    with pytest.raises(ValueError, match="evidence_prior"):
        rp.spec.ewridge("m", evidence=True, evidence_prior=-1.0, **base)
    with pytest.raises(ValueError, match="evidence_prior"):
        rp.spec.ewridge("m", evidence=True, evidence_prior=math.inf, **base)

    # huber norm refusals
    with pytest.raises(ValueError, match="huber_norm"):
        rp.spec.huber("m", huber_norm="bogus", **base)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='huber_norm="ramsay"'):
        rp.spec.huber("m", huber_norm="ramsay", huber_delta=math.inf, **base)

    # sgd refusals
    with pytest.raises(ValueError, match='schedule="optimal"'):
        rp.spec.sgd("m", schedule="optimal", l2=0.01, learning_rate=0.01, **base)
    with pytest.raises(ValueError, match="l1 \\+ l2"):
        rp.spec.sgd("m", schedule="optimal", **base)
    with pytest.raises(ValueError, match="l1"):
        rp.spec.sgd("m", l1=-0.01, **base)
    with pytest.raises(ValueError, match="l1"):
        rp.spec.sgd("m", l1=math.inf, **base)
    with pytest.raises(ValueError, match='invariant=True applies only under loss="squared"'):
        rp.spec.sgd("m", loss="huber", invariant=True, **base)
    with pytest.raises(ValueError, match='schedule="adagrad"'):
        rp.spec.sgd("m", schedule="adagrad", invariant=True, **base)
