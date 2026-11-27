"""`krls` against scikit-learn's ``KernelRidge`` and ``Nystroem + Ridge``.

Verifies online kernel recursive least squares across all four kernels
(``rbf``, ``laplacian``, ``poly``, ``linear``), both forgetting modes (ALD
dictionary with and without exponential decay, and sliding windows by row
count and clock span with ``closed="right"`` / ``"both"``), both
``fit_intercept=True`` and ``fit_intercept=False``, multi-target and
weighted streams, zero-weight rows, chunk invariance, mid-stream save/load,
residual diagnostics, clock-window ``max_state_bytes`` notices and refusals,
all spec refusals, and a 50-seed ``oracle_grid`` registered with the oracle
ledger.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_kernels

import ledger
from rheon import polars as rp

TIER = "essential"


def _kernel_matrix(
    x: np.ndarray,
    y: np.ndarray,
    kernel: str,
    gamma: float | None,
    degree: int | None,
    coef0: float | None,
) -> np.ndarray:
    """Kernel matrix via scikit-learn's ``pairwise_kernels`` matching ``krls`` defaults."""
    k = x.shape[1]
    kw: dict[str, Any] = {}
    if kernel in ("rbf", "laplacian"):
        kw["gamma"] = 1.0 / k if gamma is None else gamma
    elif kernel == "poly":
        kw["gamma"] = 1.0 / k if gamma is None else gamma
        kw["degree"] = 3 if degree is None else degree
        kw["coef0"] = 1.0 if coef0 is None else coef0
    return pairwise_kernels(x, y, metric=kernel, **kw)


def _nystroem_whitened_map(
    atoms: np.ndarray,
    x: np.ndarray,
    kernel: str,
    gamma: float | None,
    degree: int | None,
    coef0: float | None,
) -> np.ndarray:
    """Exact whitened RKHS coordinates ``psi(x)`` using scikit-learn's
    ``pairwise_kernels`` and eigendecomposition (exact same subspace and RKHS
    norm as ``sklearn.kernel_approximation.Nystroem``, handling rank-deficient
    kernels such as ``linear`` and ``poly`` cleanly)."""
    k_dd = _kernel_matrix(atoms, atoms, kernel, gamma, degree, coef0)
    eigvals, eigvecs = np.linalg.eigh(k_dd)
    keep = eigvals > max(float(eigvals[-1]), 0.0) * 1e-12
    norm = eigvecs[:, keep] / np.sqrt(eigvals[keep])
    k_xd = _kernel_matrix(x, atoms, kernel, gamma, degree, coef0)
    return k_xd @ norm


def _ald_online_features(
    x: np.ndarray,
    kernel: str,
    gamma: float | None,
    degree: int | None,
    coef0: float | None,
    max_atoms: int,
    ald_threshold: float,
) -> tuple[list[int], np.ndarray, list[np.ndarray]]:
    """Simulate Engel's ALD admission and return for each row ``i``:
    - ``counts[i]``: dictionary size before row ``i``
    - ``psi_arr``: whitened features at arrival (padded to final ``m``)
    - ``z_query[i]``: whitened query vector for row ``i`` in the dictionary
      before row ``i`` (length ``counts[i]``)."""
    n = x.shape[0]
    atoms: list[int] = []
    l_d = np.zeros((0, 0), dtype=float)
    counts: list[int] = []
    arrival_z: list[np.ndarray] = []
    query_z: list[np.ndarray] = []
    for i in range(n):
        counts.append(len(atoms))
        xi = x[i : i + 1]
        kxx = float(_kernel_matrix(xi, xi, kernel, gamma, degree, coef0)[0, 0])
        if not atoms:
            query_z.append(np.zeros(0, dtype=float))
            if max_atoms > 0 and kxx > 0.0:
                atoms.append(i)
                l_d = np.array([[math.sqrt(kxx)]], dtype=float)
                arrival_z.append(np.array([math.sqrt(kxx)], dtype=float))
            else:
                arrival_z.append(np.zeros(0, dtype=float))
            continue
        k_dx = _kernel_matrix(x[atoms], xi, kernel, gamma, degree, coef0)[:, 0]
        z = np.linalg.solve(l_d, k_dx)
        query_z.append(z.copy())
        delta = kxx - float(np.dot(z, z))
        if len(atoms) < max_atoms and delta > ald_threshold * kxx and delta > 0.0:
            m = len(atoms)
            l_new = np.zeros((m + 1, m + 1), dtype=float)
            l_new[:m, :m] = l_d
            l_new[m, :m] = z
            l_new[m, m] = math.sqrt(delta)
            l_d = l_new
            atoms.append(i)
            arrival_z.append(np.append(z, math.sqrt(delta)))
        else:
            arrival_z.append(z.copy())
    m_final = len(atoms)
    psi_arr = np.zeros((n, m_final), dtype=float)
    for i, z in enumerate(arrival_z):
        psi_arr[i, : len(z)] = z
    return counts, psi_arr, query_z


def _ridge_oracle_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_query: np.ndarray,
    atoms: np.ndarray,
    *,
    kernel: str,
    gamma: float | None,
    degree: int | None,
    coef0: float | None,
    ridge: float,
    fit_intercept: bool,
) -> np.ndarray:
    """Exact scikit-learn ``Ridge`` fit in the whitened RKHS basis spanned by ``atoms``."""
    z_train = _nystroem_whitened_map(atoms, x_train, kernel, gamma, degree, coef0)
    z_query = _nystroem_whitened_map(atoms, x_query, kernel, gamma, degree, coef0)
    reg = Ridge(alpha=ridge, fit_intercept=fit_intercept, solver="cholesky")
    reg.fit(z_train, y_train, sample_weight=w_train)
    return np.asarray(reg.predict(z_query), dtype=float)


def _kernel_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_query: np.ndarray,
    *,
    kernel: str,
    gamma: float | None,
    degree: int | None,
    coef0: float | None,
    ridge: float,
) -> np.ndarray:
    """Exact ``sklearn.kernel_ridge.KernelRidge`` (no intercept) on the given rows."""
    k = x_train.shape[1]
    kw: dict[str, Any] = {"alpha": ridge, "kernel": kernel}
    if kernel in ("rbf", "laplacian"):
        kw["gamma"] = 1.0 / k if gamma is None else gamma
    elif kernel == "poly":
        kw["gamma"] = 1.0 / k if gamma is None else gamma
        kw["degree"] = 3 if degree is None else degree
        kw["coef0"] = 1.0 if coef0 is None else coef0
    kr = KernelRidge(**kw)
    kr.fit(x_train, y_train, sample_weight=w_train)
    return np.asarray(kr.predict(x_query), dtype=float)


def _nonlinear_frame(n: int, k: int, seed: int, *, weighted: bool = True) -> pl.DataFrame:
    rng = np.random.default_rng(502_000 + seed)
    x = rng.uniform(-1.5, 1.5, size=(n, k))
    y0 = np.sin(x[:, 0]) + 0.5 * x[:, 1 % k] ** 2 - 0.3 * np.cos(x[:, 0] * x[:, -1])
    y0 = y0 + 0.05 * rng.standard_normal(n)
    y1 = np.cos(x[:, 0]) - 0.4 * x[:, 1 % k] + 0.05 * rng.standard_normal(n)
    w = rng.uniform(0.3, 2.0, n) if weighted else np.ones(n)
    t = np.cumsum(rng.choice([0.5, 1.0, 1.5, 2.0], size=n))
    data: dict[str, Any] = {"t": t, "w": w, "y0": y0, "y1": y1}
    for j in range(k):
        data[f"x{j}"] = x[:, j]
    return pl.DataFrame(data)


# --- Third-party oracle checks: Window mode vs KernelRidge / Nystroem+Ridge ---


class TestWindowAgainstScikitLearn:
    @pytest.mark.parametrize("kernel", ["rbf", "laplacian", "poly", "linear"])
    @pytest.mark.parametrize("fit_intercept", [False, True])
    def test_window_rows_matches_sklearn(self, kernel: str, fit_intercept: bool) -> None:
        df = _nonlinear_frame(55, k=3, seed=11)
        x = df.select(["x0", "x1", "x2"]).to_numpy()
        y = df.select(["y0", "y1"]).to_numpy()
        w = df["w"].to_numpy()
        win = 15
        ridge = 0.25
        kw: dict[str, Any] = {}
        if kernel == "poly":
            kw = {"degree": 2, "coef0": 1.0}
        spec = rp.spec.krls(
            "k",
            targets=["y0", "y1"],
            features=["x0", "x1", "x2"],
            kernel=kernel,
            ridge=ridge,
            window_rows=win,
            fit_intercept=fit_intercept,
            emit_n_atoms=True,
            weight="w",
            **kw,
        )
        out = rp.ModelBank([spec]).fit_predict(df).unnest("k")
        assert out["pred_y0"][0] is None
        assert out["withheld_reason"][0] == "not_identified"
        assert out["n_atoms"][0] == 0

        worst = 0.0
        for i in range(1, df.height):
            lo = max(0, i - win)
            assert out["n_atoms"][i] == i - lo
            if not fit_intercept:
                want = _kernel_ridge_predict(
                    x[lo:i],
                    y[lo:i],
                    w[lo:i],
                    x[i : i + 1],
                    kernel=kernel,
                    gamma=None,
                    degree=kw.get("degree"),
                    coef0=kw.get("coef0"),
                    ridge=ridge,
                )[0]
            else:
                want = _ridge_oracle_predict(
                    x[lo:i],
                    y[lo:i],
                    w[lo:i],
                    x[i : i + 1],
                    x[lo:i],
                    kernel=kernel,
                    gamma=None,
                    degree=kw.get("degree"),
                    coef0=kw.get("coef0"),
                    ridge=ridge,
                    fit_intercept=True,
                )[0]
            got = np.array([out["pred_y0"][i], out["pred_y1"][i]], dtype=float)
            worst = max(worst, float(np.max(np.abs(got - want) / (1.0 + np.abs(want)))))
        # Linear/poly with intercept on a 15-row rank-deficient kernel Gram has
        # slightly higher condition number than RBF/Laplacian; all stay tight.
        tol = 1e-10 if kernel in ("rbf", "laplacian") else 1e-8
        assert worst < tol, (kernel, fit_intercept, worst)

    @pytest.mark.parametrize("closed", ["right", "both"])
    def test_window_size_matches_sklearn_kernel_ridge(self, closed: str) -> None:
        df = _nonlinear_frame(60, k=2, seed=19)
        t = df["t"].to_numpy()
        x = df.select(["x0", "x1"]).to_numpy()
        y = df["y0"].to_numpy()
        w = df["w"].to_numpy()
        span = 6.0
        ridge = 0.5
        spec = rp.spec.krls(
            "k",
            targets=["y0"],
            features=["x0", "x1"],
            kernel="rbf",
            gamma=0.7,
            ridge=ridge,
            window_size=span,
            closed=closed,  # type: ignore[arg-type]
            fit_intercept=False,
            emit_n_atoms=True,
            clock="t",
            gap_cap=100.0,
            weight="w",
        )
        out = rp.ModelBank([spec]).fit_predict(df).unnest("k")
        worst = 0.0
        for i in range(1, df.height):
            keep = [
                j
                for j in range(i)
                if (t[i] - t[j] <= span if closed == "both" else t[i] - t[j] < span)
            ]
            assert out["n_atoms"][i] == len(keep)
            if not keep:
                assert out["pred_y0"][i] is None
                continue
            want = _kernel_ridge_predict(
                x[keep],
                y[keep],
                w[keep],
                x[i : i + 1],
                kernel="rbf",
                gamma=0.7,
                degree=None,
                coef0=None,
                ridge=ridge,
            )[0]
            got = float(out["pred_y0"][i])
            worst = max(worst, abs(got - want) / (1.0 + abs(want)))
        assert worst < 1e-12, (closed, worst)


# --- Third-party oracle checks: ALD mode vs KernelRidge & Nystroem + Ridge ----


class TestAldAgainstScikitLearn:
    @pytest.mark.parametrize("kernel", ["rbf", "laplacian", "poly", "linear"])
    def test_uncapped_ald_matches_kernel_ridge(self, kernel: str) -> None:
        """With ``ald_threshold=0.0`` and ``max_atoms >= n`` on non-duplicate
        rows (full-rank kernels ``rbf``/``laplacian``, or exact basis for
        ``poly``/``linear``), ALD KRLS with ``fit_intercept=False`` and no
        decay matches ``sklearn.kernel_ridge.KernelRidge``."""
        df = _nonlinear_frame(35, k=2, seed=23)
        x = df.select(["x0", "x1"]).to_numpy()
        y = df["y0"].to_numpy()
        w = df["w"].to_numpy()
        ridge = 0.3
        kw: dict[str, Any] = {"degree": 2, "coef0": 1.0} if kernel == "poly" else {}
        spec = rp.spec.krls(
            "k",
            targets=["y0"],
            features=["x0", "x1"],
            kernel=kernel,
            ridge=ridge,
            max_atoms=50,
            ald_threshold=0.0,
            fit_intercept=False,
            weight="w",
            **kw,
        )
        out = rp.ModelBank([spec]).fit_predict(df).unnest("k")
        worst = 0.0
        for i in range(1, df.height):
            want = _kernel_ridge_predict(
                x[:i],
                y[:i],
                w[:i],
                x[i : i + 1],
                kernel=kernel,
                gamma=None,
                degree=kw.get("degree"),
                coef0=kw.get("coef0"),
                ridge=ridge,
            )[0]
            got = float(out["pred_y0"][i])
            worst = max(worst, abs(got - want) / (1.0 + abs(want)))
        tol = 1e-10 if kernel in ("rbf", "laplacian") else 1e-8
        assert worst < tol, (kernel, worst)

    @pytest.mark.parametrize("kernel", ["rbf", "laplacian"])
    @pytest.mark.parametrize("fit_intercept", [False, True])
    @pytest.mark.parametrize("half_life", [float("inf"), 15.0])
    def test_capped_ald_matches_nystroem_ridge(
        self, kernel: str, fit_intercept: bool, half_life: float
    ) -> None:
        df = _nonlinear_frame(65, k=3, seed=31)
        x = df.select(["x0", "x1", "x2"]).to_numpy()
        y = df.select(["y0", "y1"]).to_numpy()
        w = df["w"].to_numpy()
        t = df["t"].to_numpy()
        max_atoms = 18
        nu = 1e-2
        ridge = 0.4
        spec = rp.spec.krls(
            "k",
            targets=["y0", "y1"],
            features=["x0", "x1", "x2"],
            kernel=kernel,
            ridge=ridge,
            max_atoms=max_atoms,
            ald_threshold=nu,
            fit_intercept=fit_intercept,
            emit_n_atoms=True,
            clock="t",
            gap_cap=100.0,
            half_life=half_life,
            weight="w",
        )
        out = rp.ModelBank([spec]).fit_predict(df).unnest("k")
        counts, psi_arr, query_z = _ald_online_features(x, kernel, None, None, None, max_atoms, nu)
        worst = 0.0
        for i in range(1, df.height):
            m = counts[i]
            assert out["n_atoms"][i] == m
            if math.isinf(half_life):
                w_eff = w[:i]
            else:
                w_eff = w[:i] * (0.5 ** ((t[i - 1] - t[:i]) / half_life))
            reg = Ridge(alpha=ridge, fit_intercept=fit_intercept, solver="cholesky")
            reg.fit(psi_arr[:i, :m], y[:i], sample_weight=w_eff)
            want = np.asarray(reg.predict(query_z[i].reshape(1, m)), dtype=float)[0]
            got = np.array([out["pred_y0"][i], out["pred_y1"][i]], dtype=float)
            worst = max(worst, float(np.max(np.abs(got - want) / (1.0 + np.abs(want)))))
        assert worst < 1e-11, (kernel, fit_intercept, half_life, worst)

    def test_heldout_mse_matches_batch_kernel_ridge_and_beats_linear(self) -> None:
        df = _nonlinear_frame(500, k=3, seed=42, weighted=False)
        train, test = df.head(400), df.tail(100)
        x_tr = train.select(["x0", "x1", "x2"]).to_numpy()
        y_tr = train["y0"].to_numpy()
        x_te = test.select(["x0", "x1", "x2"]).to_numpy()
        y_te = test["y0"].to_numpy()

        krls_spec = rp.spec.krls(
            "k",
            targets=["y0"],
            features=["x0", "x1", "x2"],
            kernel="rbf",
            ridge=0.05,
            max_atoms=100,
            ald_threshold=1e-3,
            fit_intercept=False,
        )
        lin_spec = rp.spec.ewridge(
            "lin",
            targets=["y0"],
            features=["x0", "x1", "x2"],
            half_life=float("inf"),
            ridge=0.05,
        )
        bank = rp.ModelBank([krls_spec, lin_spec])
        bank.fit_predict(train)
        pred_df = bank.predict(test)
        krls_pred = pred_df["k"].struct.field("pred_y0").to_numpy()
        lin_pred = pred_df["lin"].struct.field("pred_y0").to_numpy()
        batch_pred = (
            KernelRidge(alpha=0.05, kernel="rbf", gamma=1.0 / 3.0).fit(x_tr, y_tr).predict(x_te)
        )
        krls_mse = float(np.mean((krls_pred - y_te) ** 2))
        batch_mse = float(np.mean((batch_pred - y_te) ** 2))
        lin_mse = float(np.mean((lin_pred - y_te) ** 2))
        assert krls_mse <= 1.05 * batch_mse, (krls_mse, batch_mse)
        assert krls_mse < 0.05 * lin_mse, (krls_mse, lin_mse)


# --- Bank contracts: chunking, save/load, diagnostics, HeldCap, refusals ------


class TestKrlsContracts:
    @pytest.mark.parametrize(
        "kw",
        [
            {"max_atoms": 20, "emit_n_atoms": True},
            {"kernel": "laplacian", "max_atoms": 20, "half_life": 20.0},
            {"kernel": "poly", "degree": 2, "window_rows": 15, "emit_n_atoms": True},
            {"window_size": 10.0, "closed": "both", "emit_n_atoms": True},
        ],
        ids=["ald", "ald_decayed", "window_rows", "window_clock"],
    )
    def test_chunking_save_load_and_zero_weight(self, kw: dict[str, Any]) -> None:
        df = _nonlinear_frame(120, k=2, seed=55)
        w = df["w"].to_numpy().copy()
        w[[0, 7, 25, 26, 80, 119]] = 0.0
        df = df.with_columns(pl.Series("w", w))
        spec = rp.spec.krls(
            "k",
            targets=["y0"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=50.0,
            weight="w",
            **kw,
        )
        whole = rp.ModelBank([spec]).fit_predict(df)

        # Chunk invariance (chunks of 17).
        bank_c = rp.ModelBank([spec])
        parts = [bank_c.fit_predict(df.slice(i, 17)) for i in range(0, df.height, 17)]
        assert pl.concat(parts).equals(whole, null_equal=True)

        # Mid-stream save/load bit-exactness.
        b1 = rp.ModelBank([spec])
        b1.fit_predict(df.head(50))
        b2 = rp.ModelBank.load_bytes(b1.save_bytes(), specs=[spec])
        tail = df.tail(df.height - 50)
        assert b1.fit_predict(tail).equals(b2.fit_predict(tail), null_equal=True)

        # Zero-weight rows match the filtered stream on positive-weight rows.
        pos = df.filter(pl.col("w") > 0)
        out_pos = rp.ModelBank([spec]).fit_predict(pos)
        out_filtered = whole.filter(pl.col("w") > 0)
        assert out_filtered.equals(out_pos, null_equal=True)

    def test_diagnostics_embargo_label_clock_and_grid(self) -> None:
        df = _nonlinear_frame(100, k=2, seed=61)
        df = df.with_columns(lt=pl.col("t") + 2.0)

        diag = rp.spec.krls(
            "d",
            targets=["y0"],
            features=["x0", "x1"],
            max_atoms=20,
            emit_n_atoms=True,
            clock="t",
            gap_cap=50.0,
            half_life=[10.0, 40.0],
            emit_sigma=True,
            emit_zscore=True,
            emit_metrics=True,
            conformal=0.9,
            resid_quantiles=[0.5, 0.9],
            emit_autocorr=True,
            emit_drift=True,
            drift_threshold=8.0,
            emit_calibration=True,
            emit_breaks=True,
            emit_specification=True,
            emit_tails=True,
            emit_feature_health=True,
            emit_clocks=True,
            emit_target_weight=True,
            emit_selected=True,
            emit_averaged=True,
        )
        out = rp.ModelBank([diag]).fit_predict(df).unnest("d")
        assert "n_atoms@h10" in out.columns and "n_atoms@h40" in out.columns
        assert out["pred_y0__selected"].drop_nulls().len() > 50
        assert out["pred_y0__averaged"].drop_nulls().len() > 50

        # Embargo and label_clock on identical delay agree.
        emb = rp.spec.krls(
            "e",
            targets=["y0"],
            features=["x0", "x1"],
            max_atoms=20,
            clock="t",
            gap_cap=50.0,
            embargo=2.0,
        )
        lbl = rp.spec.krls(
            "e",
            targets=["y0"],
            features=["x0", "x1"],
            max_atoms=20,
            clock="t",
            gap_cap=50.0,
            label_clock="lt",
        )
        assert (
            rp.ModelBank([emb])
            .fit_predict(df)
            .equals(rp.ModelBank([lbl]).fit_predict(df), null_equal=True)
        )

    def test_clock_window_held_cap_warns_and_refuses(self) -> None:
        n = 60
        df = _nonlinear_frame(n, k=2, seed=71).with_columns(
            t=pl.Series(np.arange(n, dtype=float) * 0.01)
        )
        # Tight cap that fits ~25 rows in a 2-feature, 1-target clock window.
        spec = rp.spec.krls(
            "k",
            targets=["y0"],
            features=["x0", "x1"],
            window_size=100.0,
            clock="t",
            gap_cap=1000.0,
            max_state_bytes=5000,
        )
        bank = rp.ModelBank([spec])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bank.fit_predict(df.head(22))
        assert any("max_state_bytes" in str(w.message) for w in caught)
        with warnings.catch_warnings(record=True) as caught2:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="max_state_bytes"):
                bank.fit_predict(df.tail(n - 22))
        assert not caught2

    def test_refusals(self) -> None:
        base: dict[str, Any] = {"targets": ["y0"], "features": ["x0", "x1"]}
        cases = [
            ({"kernel": "sigmoid"}, "kernel"),
            ({"kernel": "linear", "gamma": 0.5}, "gamma"),
            ({"kernel": "linear", "degree": 2}, "degree"),
            ({"kernel": "linear", "coef0": 1.0}, "coef0"),
            ({"kernel": "rbf", "degree": 2}, "degree"),
            ({"kernel": "laplacian", "coef0": 1.0}, "coef0"),
            ({"kernel": "poly", "degree": 0}, "degree"),
            ({"kernel": "poly", "coef0": -0.5}, "coef0"),
            ({"gamma": 0.0}, "gamma"),
            ({"ridge": 0.0}, "ridge"),
            ({"ridge": -1.0}, "ridge"),
            ({"max_atoms": 0}, "max_atoms"),
            ({"ald_threshold": -0.1}, "ald_threshold"),
            ({"ald_threshold": 1.0}, "ald_threshold"),
            ({"window_rows": 0}, "window_rows"),
            ({"window_size": 0.0}, "window_size"),
            ({"window_rows": 10, "window_size": 10.0}, "window_rows"),
            ({"closed": "both"}, "closed"),
            ({"window_rows": 10, "half_life": 20.0}, "window"),
            ({"window_size": 10.0, "lam": 0.95}, "window"),
            ({"window_rows": 10, "max_atoms": 20}, "max_atoms"),
            ({"window_size": 10.0, "ald_threshold": 0.01}, "ald_threshold"),
            ({"max_state_bytes": 0}, "max_state_bytes"),
            ({"max_atoms": 5000, "max_state_bytes": 4096}, "max_state_bytes"),
            ({"coef_every": 5}, "coef_every"),
            ({"emit_se_coef": True}, "emit_se_coef"),
            ({"emit_robust_se": True}, "emit_robust_se"),
            ({"emit_influence": True}, "emit_influence"),
            ({"emit_error_inflation": True}, "emit_error_inflation"),
            ({"max_error_inflation": 2.0}, "max_error_inflation"),
        ]
        for kw, needle in cases:
            with pytest.raises(ValueError, match=needle):
                rp.spec.krls("m", **base, **kw)


# --- 50-seed oracle grid registered with the oracle ledger --------------------


GRID_SEEDS = 50


def _grid_outputs() -> dict[str, list[np.ndarray]]:
    """Run ``krls`` on the 50-seed grid's streams without calling scikit-learn."""
    ald_uncapped: list[np.ndarray] = []
    ald_decayed: list[np.ndarray] = []
    win_rows: list[np.ndarray] = []
    win_intercept: list[np.ndarray] = []
    heldout_preds: list[np.ndarray] = []
    for seed in range(GRID_SEEDS):
        df = _nonlinear_frame(45, k=2, seed=100 + seed)
        s_uncapped = rp.spec.krls(
            "k",
            targets=["y0"],
            features=["x0", "x1"],
            kernel="rbf",
            ridge=0.2,
            max_atoms=50,
            ald_threshold=0.0,
            fit_intercept=False,
            weight="w",
        )
        s_decayed = rp.spec.krls(
            "k",
            targets=["y0"],
            features=["x0", "x1"],
            kernel="laplacian",
            ridge=0.3,
            max_atoms=15,
            ald_threshold=1e-2,
            fit_intercept=True,
            clock="t",
            gap_cap=100.0,
            half_life=15.0,
            weight="w",
        )
        s_win = rp.spec.krls(
            "k",
            targets=["y0"],
            features=["x0", "x1"],
            kernel="rbf",
            ridge=0.25,
            window_rows=12,
            fit_intercept=False,
            weight="w",
        )
        s_win_int = rp.spec.krls(
            "k",
            targets=["y0"],
            features=["x0", "x1"],
            kernel="rbf",
            ridge=0.25,
            window_rows=12,
            fit_intercept=True,
            weight="w",
        )
        ald_uncapped.append(
            rp.ModelBank([s_uncapped]).fit_predict(df)["k"].struct.field("pred_y0").to_numpy()[1:]
        )
        ald_decayed.append(
            rp.ModelBank([s_decayed]).fit_predict(df)["k"].struct.field("pred_y0").to_numpy()[1:]
        )
        win_rows.append(
            rp.ModelBank([s_win]).fit_predict(df)["k"].struct.field("pred_y0").to_numpy()[1:]
        )
        win_intercept.append(
            rp.ModelBank([s_win_int]).fit_predict(df)["k"].struct.field("pred_y0").to_numpy()[1:]
        )

        # Held-out nonlinear surface (250 train, 80 test).
        full = _nonlinear_frame(330, k=3, seed=200 + seed, weighted=False)
        s_ho = rp.spec.krls(
            "k",
            targets=["y0"],
            features=["x0", "x1", "x2"],
            kernel="rbf",
            ridge=0.05,
            max_atoms=100,
            ald_threshold=1e-3,
            fit_intercept=False,
        )
        b = rp.ModelBank([s_ho])
        b.fit_predict(full.head(250))
        heldout_preds.append(b.predict(full.tail(80))["k"].struct.field("pred_y0").to_numpy())
    return {
        "ald_uncapped": ald_uncapped,
        "ald_decayed": ald_decayed,
        "win_rows": win_rows,
        "win_intercept": win_intercept,
        "heldout_preds": heldout_preds,
    }


ledger.guard("krls", _grid_outputs, oracles=["scikit-learn"])


@pytest.mark.oracle_grid(reason="50 seeds against sklearn KernelRidge and Nystroem+Ridge")
def test_krls_oracle_grid() -> None:
    got = _grid_outputs()
    err_uncapped: list[float] = []
    err_decayed: list[float] = []
    err_win: list[float] = []
    err_win_int: list[float] = []
    excess_mse: list[float] = []

    for seed in range(GRID_SEEDS):
        df = _nonlinear_frame(45, k=2, seed=100 + seed)
        x = df.select(["x0", "x1"]).to_numpy()
        y = df["y0"].to_numpy()
        w = df["w"].to_numpy()
        t = df["t"].to_numpy()

        # 1. Uncapped ALD vs KernelRidge (last 5 rows per seed).
        e_u = 0.0
        for i in range(df.height - 5, df.height):
            want = _kernel_ridge_predict(
                x[:i],
                y[:i],
                w[:i],
                x[i : i + 1],
                kernel="rbf",
                gamma=None,
                degree=None,
                coef0=None,
                ridge=0.2,
            )[0]
            e_u = max(e_u, abs(float(got["ald_uncapped"][seed][i - 1]) - want) / (1.0 + abs(want)))
        err_uncapped.append(e_u)

        # 2. Capped decayed ALD (laplacian, fit_intercept=True) vs sklearn Ridge.
        counts, psi_arr, query_z = _ald_online_features(x, "laplacian", None, None, None, 15, 1e-2)
        e_d = 0.0
        for i in range(df.height - 5, df.height):
            m = counts[i]
            w_eff = w[:i] * (0.5 ** ((t[i - 1] - t[:i]) / 15.0))
            reg = Ridge(alpha=0.3, fit_intercept=True, solver="cholesky")
            reg.fit(psi_arr[:i, :m], y[:i], sample_weight=w_eff)
            want = float(reg.predict(query_z[i].reshape(1, m))[0])
            e_d = max(e_d, abs(float(got["ald_decayed"][seed][i - 1]) - want) / (1.0 + abs(want)))
        err_decayed.append(e_d)

        # 3. Window rows (fit_intercept=False) vs KernelRidge.
        e_w = 0.0
        for i in range(df.height - 5, df.height):
            lo = i - 12
            want = _kernel_ridge_predict(
                x[lo:i],
                y[lo:i],
                w[lo:i],
                x[i : i + 1],
                kernel="rbf",
                gamma=None,
                degree=None,
                coef0=None,
                ridge=0.25,
            )[0]
            e_w = max(e_w, abs(float(got["win_rows"][seed][i - 1]) - want) / (1.0 + abs(want)))
        err_win.append(e_w)

        # 4. Window rows (fit_intercept=True) vs Nystroem+Ridge.
        e_wi = 0.0
        for i in range(df.height - 5, df.height):
            lo = i - 12
            want = _ridge_oracle_predict(
                x[lo:i],
                y[lo:i],
                w[lo:i],
                x[i : i + 1],
                x[lo:i],
                kernel="rbf",
                gamma=None,
                degree=None,
                coef0=None,
                ridge=0.25,
                fit_intercept=True,
            )[0]
            e_wi = max(
                e_wi, abs(float(got["win_intercept"][seed][i - 1]) - want) / (1.0 + abs(want))
            )
        err_win_int.append(e_wi)

        # 5. Held-out MSE vs batch KernelRidge.
        full = _nonlinear_frame(330, k=3, seed=200 + seed, weighted=False)
        tr, te = full.head(250), full.tail(80)
        batch = (
            KernelRidge(alpha=0.05, kernel="rbf", gamma=1.0 / 3.0)
            .fit(tr.select(["x0", "x1", "x2"]).to_numpy(), tr["y0"].to_numpy())
            .predict(te.select(["x0", "x1", "x2"]).to_numpy())
        )
        y_te = te["y0"].to_numpy()
        online_mse = float(np.mean((got["heldout_preds"][seed] - y_te) ** 2))
        batch_mse = float(np.mean((batch - y_te) ** 2))
        excess_mse.append(max(0.0, online_mse / batch_mse - 1.0))

    assert (
        ledger.record("krls", "ald_uncapped_vs_kernel_ridge", err_uncapped, floor=1e-12)[0] < 1e-11
    )
    assert (
        ledger.record("krls", "ald_decayed_vs_nystroem_ridge", err_decayed, floor=1e-11)[0] < 1e-10
    )
    assert ledger.record("krls", "window_rows_vs_kernel_ridge", err_win, floor=1e-12)[0] < 1e-11
    assert (
        ledger.record("krls", "window_intercept_vs_nystroem_ridge", err_win_int, floor=1e-11)[0]
        < 1e-10
    )
    assert ledger.record("krls", "heldout_excess_mse_ratio", excess_mse, floor=1e-3)[0] < 0.05
