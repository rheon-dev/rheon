"""The online VAR(p) against statsmodels: each check returns its worst error.

Run ``python compare.py`` for the table the README quotes; ``test_online_var.py``
asserts the same checks. Errors are relative to the reference's largest entry
(``max|a - b| / max|b|``) unless stated.
"""

from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
from statsmodels.regression.linear_model import WLS
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

sys.path.insert(0, str(Path(__file__).parent))
from online_var import WindowVAR, run  # noqa: E402

from data import simulate  # noqa: E402

warnings.filterwarnings("ignore")

# (T, k, p): small (a few rows past the parameter count), medium, long.
GRID = [
    (12, 2, 1),
    (40, 3, 2),
    (300, 2, 1),
    (300, 3, 2),
    (500, 5, 3),
    (3000, 4, 4),
    (3000, 2, 8),
]

# The coordinator's grid: k = 2, 5, 10 by p = 1, 2, 5, one stream of 400 rows each.
SPEC_GRID = [(400, k, p) for k in (2, 5, 10) for p in (1, 2, 5)]


def rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def lag_design(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """``Z = [1, y_{t-1}, ..., y_{t-p}]`` and ``Y = y_t`` for ``t = p..T-1``."""
    t = len(y)
    z = np.hstack([np.ones((t - p, 1))] + [y[p - 1 - j : t - 1 - j] for j in range(p)])
    return z, y[p:]


def subsets(k: int) -> list[list[int]]:
    return [list(c) for r in range(1, k) for c in itertools.combinations(range(k), r)]


def causality_pairs(k: int) -> list[tuple[list[int], list[int]]]:
    """(caused, causing) pairs: every disjoint pair of subsets up to k = 5; past it
    (3^k of them) every single pair, each variable against all the others, and
    two splits into halves and thirds."""
    if k <= 5:
        out = []
        for caused in subsets(k):
            rest = [i for i in range(k) if i not in caused]
            sizes = range(1, len(rest) + 1)
            out += [(caused, list(c)) for r in sizes for c in itertools.combinations(rest, r)]
        return out
    out = [([e], [j]) for e in range(k) for j in range(k) if j != e]
    out += [([e], [j for j in range(k) if j != e]) for e in range(k)]
    h, t3 = k // 2, k // 3
    out += [(list(range(h)), list(range(h, k))), (list(range(t3)), list(range(t3, 2 * t3)))]
    return out


def check_fit(t: int, k: int, p: int, seed: int = 1, level: float = 0.0) -> dict[str, float]:
    """params, sigma_u, sigma_u_mle, cov_params and every Granger test, no decay."""
    y, _ = simulate(t, k, p, seed=seed, level=level)
    m, _, _ = run(y, p, chunks=5)
    r = VAR(y).fit(p)
    out = {
        "params": rel(m.params(), r.params),
        "sigma_u": rel(m.sigma_u(), r.sigma_u),
        "sigma_u_mle": rel(m.sigma_u_mle(), r.sigma_u_mle),
        "cov_params": rel(m.cov_params(), r.cov_params()),
    }
    stat_err, p_err, n = 0.0, 0.0, 0
    for caused, causing in causality_pairs(k):
        for kind in ("f", "wald"):
            a = m.test_causality(caused, causing, kind=kind)
            b = r.test_causality(caused, causing, kind=kind)
            assert a["df"] == b.df or tuple(a["df"]) == tuple(b.df)
            stat_err = max(stat_err, abs(a["statistic"] / b.test_statistic - 1))
            p_err = max(p_err, abs(a["pvalue"] / b.pvalue - 1) if b.pvalue > 0 else 0.0)
            n += 1
    out["granger_stat"] = stat_err
    out["granger_pvalue"] = p_err
    out["granger_tests"] = n
    return out


def check_level(level: float, t: int = 300, k: int = 3, p: int = 3) -> dict[str, float]:
    """Series far from zero: both against statsmodels fitted to the series minus the level.

    Subtracting the level is exact here (the values and the level share an exponent
    range), and a VAR with a constant is shift-invariant (only ``c`` moves), so the
    shifted fit is the well-conditioned reference for the slopes and the tests.
    """
    y, _ = simulate(t, k, p, seed=1, level=level)
    ref = VAR(y - level).fit(p)
    raw = VAR(y).fit(p)
    m, _, _ = run(y, p)
    j = [i for i in range(k) if i != 0]
    g_ref = ref.test_causality(0, j).test_statistic
    return {
        "online_slopes": rel(m.params()[1:], ref.params[1:]),
        "statsmodels_slopes": rel(raw.params[1:], ref.params[1:]),
        "online_granger": abs(m.test_causality(0, j)["statistic"] / g_ref - 1),
        "statsmodels_granger": abs(raw.test_causality(0, j).test_statistic / g_ref - 1),
    }


def check_out_of_sample(t: int = 120, k: int = 3, p: int = 2) -> float:
    """``pred[t]`` is statsmodels' one-step forecast from a VAR fitted to ``y[:t]``."""
    y, _ = simulate(t, k, p, seed=3)
    _, pred, _ = run(y, p)
    first = p + k * p + 1  # the first row after m + 1 targets: Sxx is positive definite
    assert np.all(np.isnan(pred[:first])) and not np.any(np.isnan(pred[first:]))
    err = 0.0
    for i in range(first, t):
        f = VAR(y[:i]).fit(p).forecast(y[i - p : i], 1)[0]
        err = max(err, rel(pred[i], f))
    return err


def check_chunks(t: int = 2000, k: int = 3, p: int = 3) -> bool:
    """1 chunk and 1000 chunks give bit-identical predictions, weights and state."""
    y, _ = simulate(t, k, p, seed=4)
    a, pa, wa = run(y, p, lam=0.995, chunks=1)
    b, pb, wb = run(y, p, lam=0.995, chunks=1000)
    same = np.array_equal(pa, pb, equal_nan=True) and np.array_equal(wa, wb)
    return same and all(np.array_equal(getattr(a, f), getattr(b, f)) for f in ("sxx", "sxy"))


def check_decay(lam: float, t: int = 1500, k: int = 3, p: int = 2) -> dict[str, float]:
    """Decay and row weights against per-equation WLS with weights ``w_t lam^(T-1-t)``."""
    y, _ = simulate(t, k, p, seed=5)
    rng = np.random.default_rng(6)
    w = rng.uniform(0.0, 2.0, size=t)
    w[rng.random(t) < 0.1] = 0.0  # zero-weight rows: advance the clock, learn nothing
    w[p] = 0.0  # including the first target row
    m, _, _ = run(y, p, lam=lam, weights=w)
    z, yy = lag_design(y, p)
    ww = w[p:] * lam ** np.arange(t - p - 1, -1, -1)
    keep = ww > 0
    params = np.column_stack([WLS(yy[keep, e], z[keep], ww[keep]).fit().params for e in range(k)])
    resid = yy - z @ params
    sse = (resid * ww[:, None]).T @ resid
    return {"params": rel(m.params(), params), "sse": rel(m.sse(), sse)}


def check_missing(t: int = 600, k: int = 3, p: int = 2) -> float:
    """A non-finite row is no target and voids the next ``p`` rows' lags: OLS on the rest."""
    y, _ = simulate(t, k, p, seed=7)
    y[[50, 51, 300, 301 + p]] = np.nan
    y[400, 1] = np.inf
    m, _, _ = run(y, p)
    z, yy = lag_design(y, p)
    ok = np.all(np.isfinite(z), axis=1) & np.all(np.isfinite(yy), axis=1)
    b = np.linalg.lstsq(z[ok], yy[ok], rcond=None)[0]
    assert m.n_eff() == ok.sum()
    return rel(m.params(), b)


def check_select_order(t: int = 800, k: int = 3, pmax: int = 6) -> float:
    """``info_criteria(p)`` from one VAR(pmax)'s sums equals ``VAR.select_order(pmax).ics``."""
    y, _ = simulate(t, k, 2, seed=8)
    m, _, _ = run(y, pmax)
    ics = VAR(y).select_order(pmax).ics
    err = 0.0
    for p in range(1, pmax + 1):
        mine = m.info_criteria(p)
        for name in ("aic", "bic", "hqic", "fpe"):
            err = max(err, abs(mine[name] - ics[name][p]) / abs(ics[name][p]))
    return err


def check_bivariate(t: int = 60, p: int = 3) -> dict[str, float]:
    """``grangercausalitytests`` (one equation, k = 2) against the same sums.

    Its ``ssr_ftest`` statistic equals ``test_causality``'s F for k = 2, while its
    denominator degrees of freedom are ``df`` where ``test_causality`` uses ``2 df``:
    the same statistic, two p-values. Both directions (``1 -> 0`` holds the null).
    """
    y, _ = simulate(t, 2, p, seed=9)
    m, _, _ = run(y, p)
    var = VAR(y).fit(p)
    out: dict[str, float] = {"stat": 0.0, "pvalue": 0.0, "var_f_stat": 0.0}
    for e in (0, 1):
        g = grangercausalitytests(y[:, [e, 1 - e]], [p])[p][0]
        for kind, name in (("ssr_f", "ssr_ftest"), ("ssr_chi2", "ssr_chi2test"), ("lr", "lrtest")):
            a = m.test_causality(e, 1 - e, kind=kind)
            out["stat"] = max(out["stat"], abs(a["statistic"] / g[name][0] - 1))
            out["pvalue"] = max(out["pvalue"], abs(a["pvalue"] / g[name][1] - 1))
        g_f = g["params_ftest"]
        out["stat"] = max(out["stat"], abs(g_f[0] / g["ssr_ftest"][0] - 1))
        var_f = var.test_causality(e, [1 - e])
        gap = abs(g["ssr_ftest"][0] / var_f.test_statistic - 1)
        out["var_f_stat"] = max(out["var_f_stat"], gap)
        out[f"p_ssr_ftest_{1 - e}to{e}"] = g["ssr_ftest"][1]
        out[f"p_var_f_{1 - e}to{e}"] = var_f.pvalue
    return out


def check_irf(t: int, k: int, p: int, periods: int = 10, seed: int = 1) -> dict[str, float]:
    """Impulse responses, their asymptotic covariance and the FEVD, no decay."""
    y, _ = simulate(t, k, p, seed=seed)
    m, _, _ = run(y, p)
    r = VAR(y).fit(p)
    irf = r.irf(periods)
    return {
        "irf": rel(m.ma_rep(periods), irf.irfs),
        "orth_irf": rel(m.orth_ma_rep(periods), irf.orth_irfs),
        "irf_cov": rel(m.irf_cov(periods), irf.cov()),
        "fevd": rel(m.fevd(periods), r.fevd(periods).decomp),
    }


def run_window(
    y: np.ndarray, p: int, window: int, weights: np.ndarray | None = None
) -> tuple[WindowVAR, np.ndarray]:
    t, k = y.shape
    model = WindowVAR(k, p, window=window)
    w = np.ones(t) if weights is None else weights
    preds = np.array([model.step(y[i], w[i])[0] for i in range(t)])
    return model, preds


def check_window(t: int = 700, k: int = 3, p: int = 2, window: int = 120) -> dict[str, float]:
    """Every row's ``pred`` against ``VAR(y[i - window : i]).fit(p)``'s forecast, and
    the final fit's params, sigma_u, cov_params and Granger tests on the last window."""
    y, _ = simulate(t, k, p, seed=21)
    model, pred = run_window(y, p, window)
    first = window  # a full window: every check below is past the warm-up
    pred_err = 0.0
    for i in range(first, t):
        f = VAR(y[i - window : i]).fit(p).forecast(y[i - p : i], 1)[0]
        pred_err = max(pred_err, rel(pred[i], f))
    r = VAR(y[t - window :]).fit(p)
    g = max(
        abs(model.test_causality(e, j)["statistic"] / r.test_causality(e, j).test_statistic - 1)
        for e, j in causality_pairs(k)
    )
    return {
        "pred": pred_err,
        "params": rel(model.params(), r.params),
        "sigma_u": rel(model.sigma_u(), r.sigma_u),
        "cov_params": rel(model.cov_params(), r.cov_params()),
        "granger_stat": g,
        "n_eff": abs(model.n_eff() - (window - p)),
    }


def check_window_gaps(t: int = 900, k: int = 3, p: int = 2, window: int = 150) -> float:
    """Zero weights (also the first target) and NaN rows inside the window: WLS on
    the window's complete rows, at every 25th row past the first full window."""
    y, _ = simulate(t, k, p, seed=22)
    y[[300, 301, 600]] = np.nan
    y[640, 2] = np.inf
    rng = np.random.default_rng(23)
    w = rng.uniform(0.0, 2.0, size=t)
    w[rng.random(t) < 0.1] = 0.0
    w[p] = 0.0
    model = WindowVAR(k, p, window=window)
    err = 0.0
    for i in range(t):
        model.step(y[i], w[i])
        if i >= window and i % 25 == 0:
            z, yy = lag_design(y[i + 1 - window : i + 1], p)
            ww = w[i + 1 - window + p : i + 1]
            ok = np.all(np.isfinite(z), axis=1) & np.all(np.isfinite(yy), axis=1) & (ww > 0)
            b = np.column_stack([WLS(yy[ok, e], z[ok], ww[ok]).fit().params for e in range(k)])
            err = max(err, rel(model.params(), b))
    return err


def check_window_drift(t: int, level: float = 0.0, k: int = 3, p: int = 2, window: int = 200):
    """Down-dating's rounding over a long stream: the final fit against statsmodels on
    the last window, shifted by ``level`` (exact; a VAR with a constant is shift-invariant)."""
    y, _ = simulate(t, k, p, seed=24, level=level)
    model = WindowVAR(k, p, window=window)
    for i in range(t):
        model.step(y[i])
    r = VAR(y[t - window :] - level).fit(p)
    j = list(range(1, k))
    g_ref = r.test_causality(0, j).test_statistic
    return {
        "slopes": rel(model.params()[1:], r.params[1:]),
        "sigma_u": rel(model.sigma_u(), r.sigma_u),
        "granger_stat": abs(model.test_causality(0, j)["statistic"] / g_ref - 1),
    }


def statsmodels_causing_none_bug(t: int = 300, k: int = 3, p: int = 2) -> str:
    """``test_causality(caused)`` with ``causing=None`` and ``k > 2``."""
    y, _ = simulate(t, k, p, seed=1)
    r = VAR(y).fit(p)
    try:
        res = r.test_causality(0)
    except IndexError as exc:
        return f"IndexError: {exc}"
    full = r.test_causality(0, list(range(1, k)))
    return f"no error; df {res.df} against {full.df}"


def main() -> None:
    np.set_printoptions(precision=3)
    print("fit and Granger, no decay (relative errors):")
    for t, k, p in GRID:
        res = check_fit(t, k, p)
        print(f"  T={t:5d} k={k} p={p}: " + ", ".join(f"{a}={b:.1e}" for a, b in res.items()))
    for level in (1e2, 1e4, 1e6):
        res = check_level(level)
        print(f"level {level:.0e}: " + ", ".join(f"{a}={b:.1e}" for a, b in res.items()))
    print(f"out of sample: pred vs forecast on y[:t], max rel {check_out_of_sample():.1e}")
    print(f"chunk invariance (1 vs 1000 chunks), bit-identical: {check_chunks()}")
    for lam in (1.0, 0.99, 0.9):
        res = check_decay(lam)
        print(f"decay lam={lam}: " + ", ".join(f"{a}={b:.1e}" for a, b in res.items()))
    print(f"missing rows: params vs OLS on complete rows, rel {check_missing():.1e}")
    print(f"select_order ics (p=1..6 from one VAR(6)): max rel {check_select_order():.1e}")
    res = check_bivariate()
    print("grangercausalitytests: " + ", ".join(f"{a}={b:.3g}" for a, b in res.items()))
    print(f"statsmodels test_causality(0) with causing=None, k=3: {statsmodels_causing_none_bug()}")


if __name__ == "__main__":
    main()
