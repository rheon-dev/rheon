"""Smaller options and windowed two-sample KS feature health: third-party oracles and contracts.

Covers:
- ``kmeans`` given frozen ``centers``, ``emit_distances``, and ``emit_inertia``
  against ``sklearn.cluster.KMeans`` (``transform``, ``predict``, ``-score``).
- ``ew_class`` fixed ``priors``, non-fading ``reg_param``, and diagonal
  ``var_smoothing`` against ``sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis``
  and ``sklearn.naive_bayes.GaussianNB``.
- ``holt`` multi-horizon ``forecast_spans`` with ``start="known"`` against
  ``statsmodels.tsa.holtwinters.Holt(..., initialization_method="known")``.
- ``corrchange`` ``alpha_adjust="holm"`` and ``emit_n_flagged`` against
  ``statsmodels.stats.multitest.multipletests(..., method="holm")``.
- Standalone ``feature_ks_window`` / ``feature_ks_recent`` against
  ``scipy.stats.ks_2samp`` (plus ``river.drift.KSWIN`` reference).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest
from scipy import stats
from scipy.special import logsumexp
from sklearn.cluster import KMeans as SkKMeans
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.holtwinters import Holt as SmHolt

import ledger
from rheon import polars as rp

TIER = "mixed"
GRID_SEEDS = range(50)


# ---------------------------------------------------------------------------
# 1. kmeans: frozen centers, emit_distances, emit_inertia (50-seed oracle grid)
# ---------------------------------------------------------------------------


def _kmeans_stream(seed: int, k: int = 4, p: int = 3, n: int = 200):
    rng = np.random.default_rng(525_100 + seed)
    centers = rng.standard_normal((k, p)) * 2.5
    pick = rng.integers(k, size=n)
    x = centers[pick] + rng.standard_normal((n, p)) * 1.5
    dt = rng.uniform(0.5, 2.5, size=n)
    dt[0] = 0.0
    t = np.cumsum(dt)
    w = rng.uniform(0.5, 2.0, size=n)
    # Interleaved zero-weight rows + zero-weight first row (Rule 9).
    w[0] = 0.0
    w[rng.random(n) < 0.1] = 0.0
    return centers, x, t, dt, w


def _kmeans_grid_outputs() -> list[tuple[np.ndarray, np.ndarray, float]]:
    out = []
    for seed in GRID_SEEDS:
        centers, x, t, _, w = _kmeans_stream(seed)
        k, p = centers.shape
        df = pl.DataFrame(
            {"t": t, "w": w, **{f"x{i}": x[:, i] for i in range(p)}},
        )
        spec = rp.spec.kmeans(
            "km",
            features=[f"x{i}" for i in range(p)],
            centers=centers.tolist(),
            clock="t",
            gap_cap=100.0,
            weight="w",
            half_life=30.0,
            emit_distances=True,
            emit_inertia=True,
        )
        bank = rp.ModelBank([spec])
        res = bank.fit_predict(df).unnest("km")
        # Append one final zero-weight probe row at the last timestamp to read
        # the post-stream EW inertia without decaying or mutating it.
        probe = pl.DataFrame(
            {"t": [float(t[-1])], "w": [0.0], **{f"x{i}": [0.0] for i in range(p)}}
        )
        final_inertia = float(bank.predict(probe).unnest("km")["inertia"][0])
        dists = np.column_stack([res[f"dist_{j}"].to_numpy() for j in range(k)])
        labels = res["cluster"].to_numpy()
        out.append((dists, labels, final_inertia))
    return out


ledger.guard(
    "kmeans_centers_inertia",
    _kmeans_grid_outputs,
    oracles=["scikit-learn"],
)


@pytest.mark.oracle_grid(
    reason="50 seeds of frozen kmeans vs sklearn KMeans transform/predict/score"
)
def test_kmeans_centers_distances_inertia_matches_sklearn_over_50_seeds():
    rheon_out = _kmeans_grid_outputs()
    dist_gaps: list[float] = []
    label_mismatches: list[float] = []
    inertia_gaps: list[float] = []
    for seed, (dists, labels, final_inertia) in zip(GRID_SEEDS, rheon_out, strict=True):
        centers, x, t, _, w = _kmeans_stream(seed)
        k = centers.shape[0]
        ref = SkKMeans(n_clusters=k, init=centers, n_init=1, max_iter=1).fit(centers)
        ref.cluster_centers_ = centers.copy()
        dist_gaps.append(float(np.max(np.abs(dists - ref.transform(x)))))
        label_mismatches.append(float(np.sum(labels != ref.predict(x))))
        # Effective decayed weights at the last positive-weight timestamp (Rule 9:
        # zero-weight rows do not advance the clock or decay stored inertia).
        t_last = float(t[np.flatnonzero(w > 0.0)[-1]])
        age = np.maximum(t_last - t, 0.0)
        omega = w * np.power(0.5, age / 30.0)
        want_inertia = float(-ref.score(x, sample_weight=omega))
        inertia_gaps.append(abs(final_inertia - want_inertia) / max(want_inertia, 1e-12))

    ledger.record("kmeans_centers_inertia", "distances_abs_gap", dist_gaps, floor=1e-12)
    ledger.record(
        "kmeans_centers_inertia",
        "assignment_mismatches",
        label_mismatches,
        exact=True,
    )
    ledger.record("kmeans_centers_inertia", "inertia_rel_gap", inertia_gaps, floor=1e-12)

    assert max(dist_gaps) < 1e-10
    assert max(label_mismatches) == 0.0
    assert max(inertia_gaps) < 1e-12


# ---------------------------------------------------------------------------
# 2. ew_class: fixed priors, reg_param, var_smoothing (50-seed oracle grid)
# ---------------------------------------------------------------------------


def _class_stream(seed: int, n: int = 300, k: int = 4, ncls: int = 3):
    rng = np.random.default_rng(525_200 + seed)
    means = rng.standard_normal((ncls, k)) * 1.5
    mats = [rng.standard_normal((k, k)) * 0.4 + np.eye(k) for _ in range(ncls)]
    probs = rng.dirichlet(np.ones(ncls))
    probs = np.maximum(probs, 0.08)
    probs /= probs.sum()
    y = rng.choice(ncls, size=n, p=probs)
    x = np.array([means[c] + rng.standard_normal(k) @ mats[c] for c in y])
    xq = rng.standard_normal((15, k)) * 2.0
    return x, y, xq


def _ew_class_grid_outputs() -> list[tuple[np.ndarray, np.ndarray]]:
    classes = ["c0", "c1", "c2"]
    priors = [0.5, 0.3, 0.2]
    out = []
    for seed in GRID_SEEDS:
        x, y, xq = _class_stream(seed)
        k = x.shape[1]
        feats = [f"x{i}" for i in range(k)]
        train = pl.DataFrame(
            {"lab": [classes[c] for c in y], **{f: x[:, i] for i, f in enumerate(feats)}}
        )
        query = pl.DataFrame(
            {"lab": [None] * len(xq), **{f: xq[:, i] for i, f in enumerate(feats)}},
            schema={"lab": pl.String, **{f: pl.Float64 for f in feats}},
        )
        sp_qda = rp.spec.ew_class(
            "qda",
            features=feats,
            label="lab",
            classes=classes,
            covariance_type="full",
            priors=priors,
            reg_param=0.1,
            lam=1.0,
            min_weight=0.0,
        )
        sp_gnb = rp.spec.ew_class(
            "gnb",
            features=feats,
            label="lab",
            classes=classes,
            covariance_type="diag",
            priors=priors,
            var_smoothing=1e-2,
            lam=1.0,
            min_weight=0.0,
        )
        bank = rp.ModelBank([sp_qda, sp_gnb])
        bank.fit(train)
        pred = bank.predict(query)
        p_qda = np.column_stack([pred["qda"].struct.field(f"p_{c}").to_numpy() for c in classes])
        p_gnb = np.column_stack([pred["gnb"].struct.field(f"p_{c}").to_numpy() for c in classes])
        out.append((p_qda, p_gnb))
    return out


ledger.guard(
    "ew_class_priors_reg_vs",
    _ew_class_grid_outputs,
    oracles=["scikit-learn"],
)


@pytest.mark.oracle_grid(
    reason="50 seeds of ew_class priors/reg_param/var_smoothing vs sklearn QDA/GaussianNB"
)
def test_ew_class_priors_reg_and_var_smoothing_match_sklearn_over_50_seeds():
    rheon_out = _ew_class_grid_outputs()
    priors = np.array([0.5, 0.3, 0.2])
    qda_gaps: list[float] = []
    gnb_gaps: list[float] = []
    for seed, (got_qda, got_gnb) in zip(GRID_SEEDS, rheon_out, strict=True):
        x, y, xq = _class_stream(seed)
        qda = QuadraticDiscriminantAnalysis(priors=priors, reg_param=0.1, tol=1e-12).fit(x, y)
        gnb = GaussianNB(priors=priors, var_smoothing=1e-2).fit(x, y)
        qda_gaps.append(float(np.max(np.abs(got_qda - qda.predict_proba(xq)))))
        gnb_gaps.append(float(np.max(np.abs(got_gnb - gnb.predict_proba(xq)))))

    ledger.record("ew_class_priors_reg_vs", "qda_priors_reg_abs_gap", qda_gaps, floor=1e-12)
    ledger.record("ew_class_priors_reg_vs", "gnb_priors_vs_abs_gap", gnb_gaps, floor=1e-12)

    assert max(qda_gaps) < 1e-10
    assert max(gnb_gaps) < 1e-11


# ---------------------------------------------------------------------------
# 3. holt: forecast_spans on start="known" (50-seed oracle grid)
# ---------------------------------------------------------------------------


def _holt_stream(seed: int, n: int = 80):
    rng = np.random.default_rng(525_300 + seed)
    l0 = float(rng.uniform(-5.0, 5.0))
    b0 = float(rng.uniform(-0.3, 0.3))
    y = l0 + b0 * np.arange(1, n + 1, dtype=float) + rng.standard_normal(n) * 0.6
    return y, l0, b0


def _holt_grid_outputs() -> list[tuple[np.ndarray, np.ndarray]]:
    hl_level = 5.0
    hl_trend = 15.0
    spans = [1.0, 4.0, 10.0]
    out = []
    for seed in GRID_SEEDS:
        y, l0, b0 = _holt_stream(seed)
        df = pl.DataFrame({"y": y})
        spec = rp.spec.holt(
            "h",
            targets=["y"],
            half_life=hl_level,
            trend_half_life=hl_trend,
            start="known",
            initial_level=l0,
            initial_trend=b0,
            forecast_spans=spans,
            min_weight=0.0,
        )
        bank = rp.ModelBank([spec])
        res = bank.fit_predict(df).unnest("h")
        # Row 0..n-1 forecasts + post-stream probe row (matches statsmodels fit().forecast(h)).
        probe = bank.predict(pl.DataFrame({"y": [0.0]})).unnest("h")
        f_rows = np.column_stack([res[f"forecast_h{int(s)}_y"].to_numpy() for s in spans])
        f_probe = np.array([[float(probe[f"forecast_h{int(s)}_y"][0]) for s in spans]])
        s_rows = np.column_stack([res[f"forecast_sum_h{int(s)}_y"].to_numpy() for s in spans])
        s_probe = np.array([[float(probe[f"forecast_sum_h{int(s)}_y"][0]) for s in spans]])
        out.append((np.vstack([f_rows, f_probe]), np.vstack([s_rows, s_probe])))
    return out


ledger.guard(
    "holt_known_horizons",
    _holt_grid_outputs,
    oracles=["statsmodels"],
)


@pytest.mark.oracle_grid(
    reason="50 seeds of holt start='known' + forecast_spans vs statsmodels Holt"
)
def test_holt_known_start_forecast_spans_matches_statsmodels_over_50_seeds():
    rheon_out = _holt_grid_outputs()
    alpha = 1.0 - 0.5 ** (1.0 / 5.0)
    beta = 1.0 - 0.5 ** (1.0 / 15.0)
    spans = [1.0, 4.0, 10.0]
    f_gaps: list[float] = []
    s_gaps: list[float] = []
    for seed, (got_f, got_s) in zip(GRID_SEEDS, rheon_out, strict=True):
        y, l0, b0 = _holt_stream(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SmHolt(
                y,
                initialization_method="known",
                initial_level=l0,
                initial_trend=b0,
            ).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
        # Check row 0 (pre-first-observation: l0 + b0 * (1 + delta)) and final post-fit forecast.
        row0_f = np.array([l0 + b0 * (1.0 + d) for d in spans])
        row0_s = np.array([d * (l0 + b0 * (1.0 + 0.5 * d)) for d in spans])
        # Post-fit forecast: at probe step s = 1, span delta is (1 + delta) steps ahead.
        sm_fc = np.asarray(fit.forecast(11), dtype=float)
        final_f = np.array(
            [sm_fc[int(d)] for d in spans]
        )  # forecast(1+d) is index d (0-based = h=1)
        level_end = float(fit.level[-1])
        trend_end = float(fit.trend[-1])
        final_s = np.array([d * (level_end + trend_end * (1.0 + 0.5 * d)) for d in spans])

        # Also check every intermediate row against statsmodels' fitted level & trend!
        all_f = np.empty_like(got_f)
        all_s = np.empty_like(got_s)
        all_f[0] = row0_f
        all_s[0] = row0_s
        for i in range(1, len(y)):
            lv = float(fit.level[i - 1])
            tr = float(fit.trend[i - 1])
            all_f[i] = [lv + tr * (1.0 + d) for d in spans]
            all_s[i] = [d * (lv + tr * (1.0 + 0.5 * d)) for d in spans]
        all_f[-1] = final_f
        all_s[-1] = final_s

        f_gaps.append(float(np.max(np.abs(got_f - all_f))))
        s_gaps.append(float(np.max(np.abs(got_s - all_s))))

    ledger.record("holt_known_horizons", "forecast_abs_gap", f_gaps, floor=1e-11)
    ledger.record("holt_known_horizons", "forecast_sum_abs_gap", s_gaps, floor=1e-11)

    assert max(f_gaps) < 1e-11
    assert max(s_gaps) < 1e-11


# ---------------------------------------------------------------------------
# 4. corrchange: alpha_adjust="holm" + emit_n_flagged (50-seed oracle grid)
# ---------------------------------------------------------------------------


def _corrchange_stream(seed: int, n: int = 80, d: int = 4):
    rng = np.random.default_rng(525_400 + seed)
    half = n // 2
    # Shift 0, 1, or 2 pairs' correlation across the span to exercise Holm step-down.
    n_shift = seed % 3
    x = rng.standard_normal((n, d))
    if n_shift >= 1:
        x[half:, 1] = 0.85 * x[half:, 0] + math.sqrt(1.0 - 0.85**2) * rng.standard_normal(n - half)
    if n_shift >= 2:
        x[half:, 3] = -0.80 * x[half:, 2] + math.sqrt(1.0 - 0.80**2) * rng.standard_normal(n - half)
    return x


def _corrchange_grid_outputs() -> list[tuple[bool | None, int | None, bool | None, int | None]]:
    out = []
    for seed in GRID_SEEDS:
        x = _corrchange_stream(seed)
        d = x.shape[1]
        feats = [f"x{i}" for i in range(d)]
        df = pl.DataFrame({f: x[:, i] for i, f in enumerate(feats)})
        sp_holm = rp.spec.corrchange(
            "holm",
            features=feats,
            kind="monitor",
            window_rows=len(x),
            alpha=0.05,
            alpha_adjust="holm",
            emit_n_flagged=True,
        )
        sp_bonf = rp.spec.corrchange(
            "bonf",
            features=feats,
            kind="monitor",
            window_rows=len(x),
            alpha=0.05,
            alpha_adjust="bonferroni",
            emit_n_flagged=True,
        )
        res = rp.ModelBank([sp_holm, sp_bonf]).fit_predict(df)
        h_flag = res["holm"].struct.field("flag")[-1]
        h_nf = res["holm"].struct.field("n_flagged")[-1]
        b_flag = res["bonf"].struct.field("flag")[-1]
        b_nf = res["bonf"].struct.field("n_flagged")[-1]
        out.append((h_flag, h_nf, b_flag, b_nf))
    return out


ledger.guard(
    "corrchange_holm",
    _corrchange_grid_outputs,
    oracles=["statsmodels", "scipy"],
    bits=52,
)


def _pair_monitor_stats(x: np.ndarray, alpha: float) -> tuple[list[float], list[float]]:
    """Recompute per-pair monitor stat and scale-ratio stat (matches corrchange.rs)."""
    t, d = x.shape
    gamma = max(1, int(math.floor(math.log(t))))

    def bartlett_sd(s: np.ndarray) -> float:
        v = 0.0
        for lag in range(min(t, gamma)):
            w = 1.0 - lag / gamma
            acf = float(np.dot(s[lag:], s[: t - lag]))
            v += w * acf / t * (2.0 if lag > 0 else 1.0)
        return math.sqrt(v) if v > 0.0 else float("nan")

    def cusum(s: np.ndarray, sd: float) -> float:
        if not math.isfinite(sd) or sd <= 0.0:
            return float("nan")
        return float(np.max(np.abs(np.cumsum(s)))) / (math.sqrt(t) * sd)

    qs, ratios = [], []
    for a in range(d):
        for b in range(a + 1, d):
            xa = x[:, a] - np.mean(x[:, a])
            xb = x[:, b] - np.mean(x[:, b])
            xa /= np.sqrt(np.mean(xa**2))
            xb /= np.sqrt(np.mean(xb**2))
            rho = float(np.mean(xa * xb))
            xi = xa * xb - 0.5 * rho * (xa**2 + xb**2)
            diff = xa**2 - xb**2
            qs.append(cusum(xi, bartlett_sd(xi)))
            ratios.append(cusum(diff, bartlett_sd(diff)))
    return qs, ratios


@pytest.mark.oracle_grid(
    reason="50 seeds of corrchange Holm step-down vs statsmodels multipletests"
)
def test_corrchange_holm_matches_statsmodels_multipletests_over_50_seeds():
    rheon_out = _corrchange_grid_outputs()
    alpha = 0.05
    shift = 0.5826 / math.sqrt(80)
    flag_mismatches: list[float] = []
    nf_mismatches: list[float] = []
    for seed, (h_flag, h_nf, b_flag, b_nf) in zip(GRID_SEEDS, rheon_out, strict=True):
        x = _corrchange_stream(seed)
        qs, ratios = _pair_monitor_stats(x, alpha)
        pvals = [float(stats.kstwobign.sf(q + shift)) for q in qs]
        rej_holm, _, _, _ = multipletests(pvals, alpha=alpha, method="holm")
        # Account for scale_ratio_moved withholding on rejected pairs.
        order = np.argsort([-q for q in qs], kind="mergesort")
        m = len(qs)
        kept_holm = 0
        held = False
        for rank, idx in enumerate(order):
            if rej_holm[idx]:
                crit_r = float(stats.kstwobign.isf(alpha / (m - rank))) - shift
                if ratios[idx] > crit_r:
                    held = True
                else:
                    kept_holm += 1
        want_flag: bool | None = None if (kept_holm == 0 and held) else (kept_holm > 0)
        want_nf: int | None = None if (kept_holm == 0 and held) else kept_holm

        flag_mismatches.append(0.0 if h_flag == want_flag else 1.0)
        nf_mismatches.append(0.0 if h_nf == want_nf else 1.0)
        # Holm always flags at least as many pairs as Bonferroni on non-withheld reports.
        if h_nf is not None and b_nf is not None:
            assert h_nf >= b_nf
            assert h_flag == b_flag

    ledger.record("corrchange_holm", "holm_flag_mismatches", flag_mismatches, exact=True)
    ledger.record("corrchange_holm", "holm_n_flagged_mismatches", nf_mismatches, exact=True)

    assert max(flag_mismatches) == 0.0
    assert max(nf_mismatches) == 0.0


# ---------------------------------------------------------------------------
# 5. Standalone feature_ks_window / feature_ks_recent (50-seed oracle grid)
# ---------------------------------------------------------------------------


def _ks_stream(seed: int, n: int = 120):
    rng = np.random.default_rng(525_500 + seed)
    half = n // 2
    x0 = np.concatenate([rng.standard_normal(half), rng.standard_normal(n - half) * 1.4 + 0.6])
    # Ties on half the seeds for x1.
    x1 = np.concatenate([rng.standard_normal(half), rng.standard_normal(n - half) - 0.5])
    if seed % 2 == 1:
        x1 = np.round(x1, 1)
    y = 1.5 * x0 - 0.7 * x1 + rng.standard_normal(n) * 0.5
    w = np.ones(n, dtype=float)
    w[0] = 0.0
    w[rng.random(n) < 0.1] = 0.0
    return x0, x1, y, w


def _ks_grid_outputs() -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    out = []
    for seed in GRID_SEEDS:
        x0, x1, y, w = _ks_stream(seed)
        df = pl.DataFrame({"x0": x0, "x1": x1, "y": y, "w": w})
        # Unequal split (60, 20) and equal split (40, 20), standalone under half_life=inf!
        sp_uneq = rp.spec.ewridge(
            "ks_u",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            weight="w",
            feature_ks_window=60,
            feature_ks_recent=20,
        )
        sp_eq = rp.spec.ewridge(
            "ks_e",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            weight="w",
            feature_ks_window=40,
            feature_ks_recent=20,
        )
        res = rp.ModelBank([sp_uneq, sp_eq]).fit_predict(df)
        u_stat = np.column_stack(
            [
                res["ks_u"].struct.field("ks_stat_x0").to_numpy(),
                res["ks_u"].struct.field("ks_stat_x1").to_numpy(),
            ]
        )
        u_p = np.column_stack(
            [
                res["ks_u"].struct.field("ks_p_x0").to_numpy(),
                res["ks_u"].struct.field("ks_p_x1").to_numpy(),
            ]
        )
        e_stat = np.column_stack(
            [
                res["ks_e"].struct.field("ks_stat_x0").to_numpy(),
                res["ks_e"].struct.field("ks_stat_x1").to_numpy(),
            ]
        )
        e_p = np.column_stack(
            [
                res["ks_e"].struct.field("ks_p_x0").to_numpy(),
                res["ks_e"].struct.field("ks_p_x1").to_numpy(),
            ]
        )
        out.append((u_stat, u_p, e_stat, e_p))
    return out


ledger.guard(
    "feature_ks",
    _ks_grid_outputs,
    oracles=["scipy"],
    bits=52,
)


@pytest.mark.oracle_grid(
    reason="50 seeds of standalone feature_ks_window/recent vs scipy.stats.ks_2samp"
)
def test_feature_ks_matches_scipy_exact_over_50_seeds():
    rheon_out = _ks_grid_outputs()
    stat_gaps: list[float] = []
    p_gaps: list[float] = []
    for seed, (u_stat, u_p, e_stat, e_p) in zip(GRID_SEEDS, rheon_out, strict=True):
        x0, x1, _, w = _ks_stream(seed)
        cols = [x0, x1]
        sg, pg = 0.0, 0.0
        for window, recent, got_s, got_p in ((60, 20, u_stat, u_p), (40, 20, e_stat, e_p)):
            for fi, col in enumerate(cols):
                ring: list[float] = []
                for row_i in range(len(col)):
                    if len(ring) < window:
                        assert np.isnan(got_s[row_i, fi])
                        assert np.isnan(got_p[row_i, fi])
                    else:
                        # Spot-check every 11th row (and final row) for speed across 50 seeds.
                        if row_i % 11 == 0 or row_i == len(col) - 1:
                            res = stats.ks_2samp(ring[:-recent], ring[-recent:], method="exact")
                            sg = max(sg, abs(float(got_s[row_i, fi]) - float(res.statistic)))
                            pg = max(pg, abs(float(got_p[row_i, fi]) - float(res.pvalue)))
                    if w[row_i] > 0.0:
                        ring.append(float(col[row_i]))
                        if len(ring) > window:
                            ring.pop(0)
        stat_gaps.append(sg)
        p_gaps.append(pg)

    ledger.record("feature_ks", "ks_stat_abs_gap", stat_gaps, floor=1e-12)
    ledger.record("feature_ks", "ks_p_exact_abs_gap", p_gaps, floor=1e-12)

    assert max(stat_gaps) < 1e-12
    assert max(p_gaps) < 1e-12


# ---------------------------------------------------------------------------
# Fast contract, chunk-invariance, save/load, zero-weight, and refusal tests
# ---------------------------------------------------------------------------


def _assert_frame_bit_equal(a: pl.DataFrame, b: pl.DataFrame) -> None:
    assert a.columns == b.columns
    assert a.height == b.height
    for col in a.columns:
        sa, sb = a[col], b[col]
        if isinstance(sa.dtype, pl.Struct):
            _assert_frame_bit_equal(sa.struct.unnest(), sb.struct.unnest())
        elif sa.dtype == pl.Float64:
            na = sa.to_numpy(allow_copy=True)
            nb = sb.to_numpy(allow_copy=True)
            assert np.array_equal(na.view(np.uint64), nb.view(np.uint64)), (
                f"float bits differ in {col}"
            )
        else:
            assert sa.equals(sb, null_equal=True), f"values differ in {col}"


def test_chunk_invariance_1_vs_1000_chunks_and_save_load_all_five_features():
    rng = np.random.default_rng(77)
    n = 1_000
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.2 * x0 - 0.5 * x1 + rng.standard_normal(n) * 0.3
    lab = ["pos" if v >= 0.0 else "neg" for v in y]
    w = rng.uniform(0.5, 2.0, size=n)
    w[0] = 0.0
    w[17::23] = 0.0
    df = pl.DataFrame({"x0": x0, "x1": x1, "x2": x2, "y": y, "lab": lab, "w": w})

    specs = [
        rp.spec.kmeans(
            "km",
            features=["x0", "x1"],
            centers=[[-1.0, 0.0], [1.0, 0.0]],
            half_life=30.0,
            weight="w",
            emit_distances=True,
            emit_inertia=True,
            max_rows_between_coefs=1,
        ),
        rp.spec.ew_class(
            "cls_reg",
            features=["x0", "x1"],
            label="lab",
            classes=["neg", "pos"],
            priors={"neg": 0.4, "pos": 0.6},
            reg_param=0.15,
            half_life=30.0,
            weight="w",
            max_rows_between_coefs=1,
        ),
        rp.spec.ew_class(
            "cls_vs",
            features=["x0", "x1"],
            label="lab",
            classes=["neg", "pos"],
            covariance_type="diag",
            priors=[0.35, 0.65],
            var_smoothing=1e-2,
            half_life=30.0,
            weight="w",
            max_rows_between_coefs=1,
        ),
        rp.spec.holt(
            "hl",
            targets=["y"],
            half_life=10.0,
            start="known",
            initial_level=0.0,
            initial_trend=0.01,
            forecast_spans=[1.0, 5.0],
            weight="w",
            min_weight=0.0,
            max_rows_between_coefs=1,
        ),
        rp.spec.corrchange(
            "cc",
            features=["x0", "x1", "x2"],
            kind="monitor",
            window_rows=25,
            alpha_adjust="holm",
            emit_n_flagged=True,
            weight="w",
        ),
        rp.spec.ewridge(
            "ks",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            weight="w",
            feature_ks_window=30,
            feature_ks_recent=10,
            max_rows_between_coefs=1,
        ),
    ]

    # 1 chunk
    whole = rp.ModelBank(specs).fit_predict(df)

    # 1,000 single-row chunks
    bank_1k = rp.ModelBank(specs)
    parts = [bank_1k.fit_predict(df.slice(i, 1)) for i in range(n)]
    chunked = pl.concat(parts, how="vertical")
    _assert_frame_bit_equal(whole, chunked)

    # Save / load mid-stream at row 371 -> bit-identical tail
    bank_a = rp.ModelBank(specs)
    head = bank_a.fit_predict(df.slice(0, 371))
    saved = bank_a.save_bytes()
    bank_b = rp.ModelBank.load_bytes(saved)
    tail = bank_b.fit_predict(df.slice(371, n - 371))
    _assert_frame_bit_equal(whole, pl.concat([head, tail], how="vertical"))


def test_defaults_unchanged_zero_bits_moved():
    rng = np.random.default_rng(19)
    n = 120
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 0.5 * x0 + 0.3 * x1 + rng.standard_normal(n) * 0.2
    lab = ["a" if v < 0 else "b" for v in y]
    df = pl.DataFrame({"x0": x0, "x1": x1, "x2": x2, "y": y, "lab": lab})

    # Verify default specs omit all new optional output fields.
    km = rp.spec.kmeans("km", features=["x0", "x1"], n_clusters=2, half_life=20.0, warm_rows=10)
    assert "dist_0" not in rp.spec.output_fields(km)
    assert "inertia" not in rp.spec.output_fields(km)

    cc = rp.spec.corrchange("cc", features=["x0", "x1", "x2"], window_rows=20)
    assert "n_flagged" not in rp.spec.output_fields(cc)

    ew = rp.spec.ewridge("ew", targets=["y"], features=["x0", "x1"], half_life=20.0)
    assert "ks_stat_x0" not in rp.spec.output_fields(ew)

    # Standalone feature_ks never moves the fit fields (diagnostic isolation).
    ew_ks = rp.spec.ewridge(
        "ew",
        targets=["y"],
        features=["x0", "x1"],
        half_life=20.0,
        feature_ks_window=30,
        feature_ks_recent=10,
    )
    out_plain = rp.ModelBank([ew]).fit_predict(df).unnest("ew")
    out_ks = rp.ModelBank([ew_ks]).fit_predict(df).unnest("ew")
    for col in out_plain.columns:
        _assert_frame_bit_equal(out_plain.select(col), out_ks.select(col))


def test_ew_class_weighted_gnb_decayed_reg_param_tied_and_dict_priors():
    rng = np.random.default_rng(42)
    n, k, ncls = 200, 3, 3
    classes = ["low", "mid", "high"]
    x, y, xq = _class_stream(12, n=n, k=k, ncls=ncls)
    w = rng.uniform(0.5, 2.0, size=n)
    feats = [f"x{i}" for i in range(k)]
    train = pl.DataFrame(
        {"lab": [classes[c] for c in y], "w": w, **{f: x[:, i] for i, f in enumerate(feats)}}
    )
    query = pl.DataFrame(
        {
            "lab": [None] * len(xq),
            "w": [0.0] * len(xq),
            **{f: xq[:, i] for i, f in enumerate(feats)},
        },
        schema={"lab": pl.String, "w": pl.Float64, **{f: pl.Float64 for f in feats}},
    )

    # 1. Dict priors match list priors bit-for-bit.
    p_dict = {"low": 0.2, "mid": 0.3, "high": 0.5}
    p_list = [0.2, 0.3, 0.5]
    sp_d = rp.spec.ew_class(
        "d",
        features=feats,
        label="lab",
        classes=classes,
        priors=p_dict,
        reg_param=0.2,
        lam=1.0,
        min_weight=0.0,
    )
    sp_l = rp.spec.ew_class(
        "l",
        features=feats,
        label="lab",
        classes=classes,
        priors=p_list,
        reg_param=0.2,
        lam=1.0,
        min_weight=0.0,
    )
    bank = rp.ModelBank([sp_d, sp_l])
    bank.fit(train)
    pr = bank.predict(query)
    _assert_frame_bit_equal(pr["d"].struct.unnest(), pr["l"].struct.unnest())

    # 2. Weighted GaussianNB matches when sklearn gets rheon's weighted eps.
    vs = 0.05
    sp_w = rp.spec.ew_class(
        "gw",
        features=feats,
        label="lab",
        classes=classes,
        covariance_type="diag",
        priors=p_list,
        var_smoothing=vs,
        lam=1.0,
        weight="w",
        min_weight=0.0,
    )
    bw = rp.ModelBank([sp_w])
    bw.fit(train)
    got_w = np.column_stack(
        [bw.predict(query)["gw"].struct.field(f"p_{c}").to_numpy() for c in classes]
    )
    w_mean = (w[:, None] * x).sum(axis=0) / w.sum()
    w_var = (w[:, None] * (x - w_mean) ** 2).sum(axis=0) / w.sum()
    eps_rheon = vs * float(np.max(w_var))
    vs_sk = eps_rheon / float(np.max(np.var(x, axis=0)))
    gnb = GaussianNB(priors=np.array(p_list), var_smoothing=vs_sk).fit(x, y, sample_weight=w)
    assert np.max(np.abs(got_w - gnb.predict_proba(xq))) < 1e-12

    # 3. Decayed full + tied + diag reg_param against two-pass weighted definition.
    dt = rng.uniform(0.5, 2.0, size=n)
    dt[0] = 0.0
    t = np.cumsum(dt)
    w_z = w.copy()
    w_z[0] = 0.0
    w_z[10::15] = 0.0
    train_d = pl.DataFrame(
        {
            "t": t,
            "lab": [classes[c] for c in y],
            "w": w_z,
            **{f: x[:, i] for i, f in enumerate(feats)},
        }
    )
    query_d = pl.DataFrame(
        {
            "t": [float(t[-1])] * len(xq),
            "lab": [None] * len(xq),
            "w": [0.0] * len(xq),
            **{f: xq[:, i] for i, f in enumerate(feats)},
        },
        schema={
            "t": pl.Float64,
            "lab": pl.String,
            "w": pl.Float64,
            **{f: pl.Float64 for f in feats},
        },
    )
    reg = 0.2
    sp_full = rp.spec.ew_class(
        "f",
        features=feats,
        label="lab",
        classes=classes,
        covariance_type="full",
        priors=p_list,
        reg_param=reg,
        clock="t",
        gap_cap=100.0,
        half_life=40.0,
        weight="w",
        min_weight=0.0,
    )
    sp_tied = rp.spec.ew_class(
        "ti",
        features=feats,
        label="lab",
        classes=classes,
        covariance_type="tied",
        priors=p_list,
        reg_param=reg,
        clock="t",
        gap_cap=100.0,
        half_life=40.0,
        weight="w",
        min_weight=0.0,
    )
    sp_diag = rp.spec.ew_class(
        "dg",
        features=feats,
        label="lab",
        classes=classes,
        covariance_type="diag",
        priors=p_list,
        reg_param=reg,
        clock="t",
        gap_cap=100.0,
        half_life=40.0,
        weight="w",
        min_weight=0.0,
    )
    bd = rp.ModelBank([sp_full, sp_tied, sp_diag])
    bd.fit(train_d)
    prd = bd.predict(query_d)
    got_f = np.column_stack([prd["f"].struct.field(f"p_{c}").to_numpy() for c in classes])
    got_t = np.column_stack([prd["ti"].struct.field(f"p_{c}").to_numpy() for c in classes])
    got_dg = np.column_stack([prd["dg"].struct.field(f"p_{c}").to_numpy() for c in classes])

    omega = w_z * np.power(0.5, (t[-1] - t) / 40.0)
    means_c, covs_c, ws_c = [], [], []
    for c in range(ncls):
        idx = (y == c) & (omega > 0)
        oc = omega[idx]
        wc = oc.sum()
        mc = (oc[:, None] * x[idx]).sum(axis=0) / wc
        xc = x[idx] - mc
        sc = (xc * oc[:, None]).T @ xc / wc
        means_c.append(mc)
        covs_c.append(sc)
        ws_c.append(wc)
    w_tot = sum(ws_c)
    s_pool = sum((ws_c[c] / w_tot) * covs_c[c] for c in range(ncls))
    m_tied = (1.0 - reg) * s_pool + reg * np.eye(k)

    for qi, qx in enumerate(xq):
        ll_f, ll_t, ll_d = [], [], []
        for c in range(ncls):
            dvec = qx - means_c[c]
            mc = (1.0 - reg) * covs_c[c] + reg * np.eye(k)
            ll_f.append(
                math.log(p_list[c])
                - 0.5 * (np.linalg.slogdet(mc)[1] + dvec @ np.linalg.solve(mc, dvec))
            )
            ll_t.append(
                math.log(p_list[c])
                - 0.5 * (np.linalg.slogdet(m_tied)[1] + dvec @ np.linalg.solve(m_tied, dvec))
            )
            vd = (1.0 - reg) * np.diag(covs_c[c]) + reg
            ll_d.append(math.log(p_list[c]) - 0.5 * float(np.sum(np.log(vd) + dvec**2 / vd)))
        for got_row, lls in ((got_f[qi], ll_f), (got_t[qi], ll_t), (got_dg[qi], ll_d)):
            arr = np.asarray(lls)
            want = np.exp(arr - logsumexp(arr))
            assert np.max(np.abs(got_row - want)) < 1e-12


def test_corrchange_holm_strictly_more_flagged_than_bonferroni_and_sequential_window_scalar():
    # Construct a span where Holm flags at least as many pairs as Bonferroni
    # (and verify sequential, window, and scalar modes with emit_n_flagged).
    rng = np.random.default_rng(101)
    n, d = 160, 4
    feats = [f"x{i}" for i in range(d)]
    x = rng.standard_normal((n, d))
    x[80:, 1] = 0.85 * x[80:, 0] + math.sqrt(1 - 0.85**2) * rng.standard_normal(80)
    x[80:, 3] = 0.75 * x[80:, 2] + math.sqrt(1 - 0.75**2) * rng.standard_normal(80)
    df = pl.DataFrame({f: x[:, i] for i, f in enumerate(feats)})

    specs = [
        rp.spec.corrchange(
            "seq_h",
            features=feats,
            kind="sequential",
            window_rows=40,
            monitor_rows=80,
            alpha_adjust="holm",
            emit_n_flagged=True,
        ),
        rp.spec.corrchange(
            "seq_b",
            features=feats,
            kind="sequential",
            window_rows=40,
            monitor_rows=80,
            alpha_adjust="bonferroni",
            emit_n_flagged=True,
        ),
        rp.spec.corrchange(
            "win", features=feats, kind="window", window_rows=20, crit=0.5, emit_n_flagged=True
        ),
        rp.spec.corrchange(
            "sc", features=feats, kind="monitor", window_rows=40, scalar=True, emit_n_flagged=True
        ),
    ]
    out = rp.ModelBank(specs).fit_predict(df)
    # Sequential: Holm n_flagged >= Bonferroni n_flagged on every flagged row.
    sh = out["seq_h"].struct.unnest()
    sb = out["seq_b"].struct.unnest()
    assert sh["flag"].equals(sb["flag"], null_equal=True)
    for nf_h, nf_b in zip(sh["n_flagged"].to_list(), sb["n_flagged"].to_list(), strict=True):
        if nf_h is not None and nf_b is not None:
            assert nf_h >= nf_b
    # Window & scalar emit 0/1 matching flag.
    for col in ("win", "sc"):
        sub = out[col].struct.unnest()
        for fl, nf in zip(sub["flag"].to_list(), sub["n_flagged"].to_list(), strict=True):
            if fl is None:
                assert nf is None
            else:
                assert nf == (1 if fl else 0)


def test_feature_ks_asymptotic_together_with_feature_health_and_river_kswin_reference():
    from river.drift import KSWIN

    # 1. Works together with emit_feature_health (finite half_life).
    rng = np.random.default_rng(9)
    n = 150
    x = np.concatenate([rng.standard_normal(75), rng.standard_normal(75) + 3.0])
    y = 2.0 * x + rng.standard_normal(n) * 0.2
    df = pl.DataFrame({"x": x, "y": y})
    sp = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x"],
        half_life=20.0,
        emit_feature_health=True,
        feature_ks_window=60,
        feature_ks_recent=20,
    )
    out = rp.ModelBank([sp]).fit_predict(df).unnest("m")
    assert out["spread_ratio_x"].drop_nulls().len() > 0
    assert out["ks_stat_x"].drop_nulls().len() == n - 60
    # After the +3 shift fills the recent 20 rows, KS p-value collapses near 0.
    assert float(out["ks_p_x"][95]) < 1e-6

    # 2. Asymptotic branch (n1 * n2 > 250_000): window=1200, recent=400 -> n1*n2 = 320_000.
    x_big = np.concatenate([rng.standard_normal(800), rng.standard_normal(500) + 0.35])
    y_big = x_big + rng.standard_normal(len(x_big)) * 0.5
    df_big = pl.DataFrame({"x": x_big, "y": y_big})
    sp_big = rp.spec.ewridge(
        "mb",
        targets=["y"],
        features=["x"],
        half_life=float("inf"),
        feature_ks_window=1200,
        feature_ks_recent=400,
    )
    out_big = rp.ModelBank([sp_big]).fit_predict(df_big).unnest("mb")
    got_s = float(out_big["ks_stat_x"][-1])
    got_p = float(out_big["ks_p_x"][-1])
    ref = stats.ks_2samp(x_big[-1201:-401], x_big[-401:-1], method="asymp")
    assert abs(got_s - float(ref.statistic)) < 1e-12
    assert abs(got_p - float(ref.pvalue)) < 1e-3

    # 3. river KSWIN reference: both detect the strong shift (+3 sigma).
    kswin = KSWIN(alpha=0.005, window_size=60, stat_size=20, seed=42)
    river_detected = False
    for v in x:
        kswin.update(float(v))
        if kswin.drift_detected:
            river_detected = True
            break
    assert river_detected
    assert float(out["ks_p_x"][95]) < 0.005


def test_refusals_for_all_new_options():
    # kmeans centers/learn refusals
    with pytest.raises(ValueError, match="n_clusters is required"):
        rp.spec.kmeans("k", features=["x0", "x1"], half_life=10.0)
    with pytest.raises(ValueError, match="learn=True cannot be set with centers"):
        rp.spec.kmeans(
            "k", features=["x0", "x1"], centers=[[0.0, 0.0], [1.0, 1.0]], learn=True, half_life=10.0
        )
    with pytest.raises(ValueError, match="learn=False requires centers"):
        rp.spec.kmeans("k", features=["x0", "x1"], n_clusters=2, learn=False, half_life=10.0)
    with pytest.raises(ValueError, match="centers has 1 rows, expected n_clusters"):
        rp.spec.kmeans(
            "k", features=["x0", "x1"], n_clusters=2, centers=[[0.0, 0.0]], half_life=10.0
        )
    with pytest.raises(ValueError, match="one coordinate per feature"):
        rp.spec.kmeans("k", features=["x0", "x1"], centers=[[0.0], [1.0]], half_life=10.0)
    with pytest.raises(ValueError, match="warm_rows applies only when fitting centers"):
        rp.spec.kmeans(
            "k",
            features=["x0", "x1"],
            centers=[[0.0, 0.0], [1.0, 1.0]],
            warm_rows=10,
            half_life=10.0,
        )
    with pytest.raises(ValueError, match="scale_floor"):
        rp.spec.kmeans(
            "k",
            features=["x0", "x1"],
            centers=[[0.0, 0.0], [1.0, 1.0]],
            scale_floor=0.1,
            half_life=10.0,
        )

    # ew_class priors/reg_param/var_smoothing refusals
    with pytest.raises(ValueError, match="precision_prior is required"):
        rp.spec.ew_class("c", features=["x0"], label="y", classes=["a", "b"], half_life=10.0)
    with pytest.raises(ValueError, match="priors keys must match classes"):
        rp.spec.ew_class(
            "c",
            features=["x0"],
            label="y",
            classes=["a", "b"],
            precision_prior=0.1,
            priors={"a": 0.5},
            half_life=10.0,
        )
    with pytest.raises(ValueError, match="priors must sum to 1"):
        rp.spec.ew_class(
            "c",
            features=["x0"],
            label="y",
            classes=["a", "b"],
            precision_prior=0.1,
            priors=[0.3, 0.3],
            half_life=10.0,
        )
    with pytest.raises(ValueError, match="reg_param must be finite and between 0 and 1"):
        rp.spec.ew_class(
            "c", features=["x0"], label="y", classes=["a", "b"], reg_param=1.5, half_life=10.0
        )
    with pytest.raises(ValueError, match='covariance_type="diag"'):
        rp.spec.ew_class(
            "c",
            features=["x0"],
            label="y",
            classes=["a", "b"],
            covariance_type="full",
            var_smoothing=1e-9,
            half_life=10.0,
        )
    with pytest.raises(ValueError, match="var_smoothing and reg_param cannot both be set"):
        rp.spec.ew_class(
            "c",
            features=["x0"],
            label="y",
            classes=["a", "b"],
            covariance_type="diag",
            reg_param=0.1,
            var_smoothing=1e-9,
            half_life=10.0,
        )

    # holt forecast_spans still refused on seasonal, damped, and heuristic
    with pytest.raises(ValueError, match="forecast_spans"):
        rp.spec.holt("h", targets=["y"], half_life=5.0, start="heuristic", forecast_spans=[2.0])
    with pytest.raises(ValueError, match="forecast_spans"):
        rp.spec.holt(
            "h",
            targets=["y"],
            half_life=5.0,
            start="known",
            initial_level=0.0,
            initial_trend=0.0,
            damping_half_life=10.0,
            forecast_spans=[2.0],
        )

    # corrchange holm refused on window
    with pytest.raises(ValueError, match="alpha_adjust"):
        rp.spec.corrchange(
            "cc", features=["x0", "x1"], kind="window", window_rows=10, alpha_adjust="holm"
        )

    # feature_ks_window / feature_ks_recent refusals
    with pytest.raises(ValueError, match="feature_ks_recent"):
        rp.spec.ewridge("m", targets=["y"], features=["x0"], half_life=10.0, feature_ks_window=20)
    with pytest.raises(ValueError, match="feature_ks_window"):
        rp.spec.ewridge("m", targets=["y"], features=["x0"], half_life=10.0, feature_ks_recent=5)
    with pytest.raises(ValueError, match="must be greater than feature_ks_recent"):
        rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            half_life=10.0,
            feature_ks_window=10,
            feature_ks_recent=10,
        )
    with pytest.raises(ValueError, match="at least 1"):
        rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            half_life=10.0,
            feature_ks_window=10,
            feature_ks_recent=0,
        )
    with pytest.raises(ValueError, match="per-group cap"):
        rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=10.0,
            feature_ks_window=20_000,
            feature_ks_recent=100,
        )
    with pytest.raises(ValueError, match="has no features"):
        rp.spec.holt("h", targets=["y"], half_life=10.0, feature_ks_window=20, feature_ks_recent=5)
    with pytest.raises(ValueError, match="feature_ks_window and feature_ks_recent do not apply"):
        rp.spec.kmeans(
            "k",
            features=["x0", "x1"],
            n_clusters=2,
            half_life=10.0,
            feature_ks_window=20,
            feature_ks_recent=5,
        )
