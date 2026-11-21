"""Oracle search C12: one-case checks of third-party candidates against the paper's definition.

Run: python scripts/oracle_search.py [check ...]   (no argument: every check)
Each check builds ONE small seeded case, writes the definition by hand from the paper,
and prints the candidate's version, licence and the largest gap.  Needs the candidates in
the venv (see docs/records/ORACLE-SEARCH-2026-11-21.md); a missing one prints SKIP.
"""
from __future__ import annotations

import sys
from importlib import metadata
from math import gamma, pi, sqrt

import numpy as np

CHECKS: dict = {}


def check(fn):
    CHECKS[fn.__name__] = fn
    return fn


def ver(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def lic(dist: str) -> str:
    try:
        m = metadata.metadata(dist)
        if m.get("License-Expression"):
            return m["License-Expression"]
        cl = [c.split("::")[-1].strip() for c in (m.get_all("Classifier") or []) if c.startswith("License ::")]
        return cl[0] if cl else ((m.get("License") or "").splitlines() or ["see PyPI"])[0][:30]
    except metadata.PackageNotFoundError:
        return "?"


def report(task, quantity, dist, gaps: dict, case: str):
    print(f"[{task}] {quantity} | {dist} {ver(dist)} ({lic(dist)}) | case: {case}")
    for k, v in gaps.items():
        print(f"    {k}: {v:.3e}" if isinstance(v, float) else f"    {k}: {v}")


def rel(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(b), 1e-300)


# ---------------------------------------------------------------- definitions (by hand)
MU1 = sqrt(2 / pi)
MU43 = 2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)


def rv_def(r):
    return float(np.sum(r**2))


def bpv_def(r):  # Barndorff-Nielsen & Shephard 2004: mu1^-2 sum |r_i||r_{i-1}|
    return float(MU1**-2 * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))


def tpq_def(r):  # BNS 2006, M/(M-2) small-sample factor, M = n, unit interval
    n = len(r)
    a = np.abs(r) ** (4 / 3)
    return float(n * MU43**-3 * n / (n - 2) * np.sum(a[2:] * a[1:-1] * a[:-2]))


def medrv_def(r):  # Andersen, Dobrev, Schaumburg 2012
    n = len(r)
    a = np.abs(r)
    med = np.median(np.stack([a[:-2], a[1:-1], a[2:]]), axis=0)
    return float(pi / (6 - 4 * sqrt(3) + pi) * n / (n - 2) * np.sum(med**2))


def tsrv_def(r, K):  # Zhang, Mykland, Ait-Sahalia 2005: slow RV averaged over K grids, minus noise bias
    n = len(r)
    p = np.concatenate([[0.0], np.cumsum(r)])
    slow = np.mean([np.sum(np.diff(p[k::K]) ** 2) for k in range(K)])
    nbar = (n - K + 1) / K
    return float(slow - nbar / n * rv_def(r))


def hy_def(tx, x, ty, y):  # Hayashi & Yoshida 2005: sum of products of overlapping increments
    s = 0.0
    for i in range(1, len(tx)):
        for j in range(1, len(ty)):
            if tx[i - 1] < ty[j] and ty[j - 1] < tx[i]:
                s += (x[i] - x[i - 1]) * (y[j] - y[j - 1])
    return s


# ---------------------------------------------------------------- checks
@check
def realized_mfe():
    import mfe.realized as m

    rng = np.random.default_rng(1)
    r = rng.standard_normal(500) * 1e-3
    v = lambda res: getattr(res, "value", res)
    g = {
        "RV": rel(v(m.realized_variance(r)), rv_def(r)),
        "BPV": rel(v(m.realized_bipower_variation(r)), bpv_def(r)),
        "MedRV": rel(v(m.realized_med_variance(r)), medrv_def(r)),
        "TPQ": rel(v(m.realized_tripower_quarticity(r)), tpq_def(r)),
        "TSRV(K=7)": rel(m.tsrv(r, K=7).tsrv, tsrv_def(r, 7)),
    }
    report("499", "realized measures", "mfe-toolbox", g, "500 iid N(0,1e-3^2) returns, rel. gap")


@check
def hy_candidates():
    rng = np.random.default_rng(2)
    tx = np.sort(np.concatenate([[0], rng.uniform(0, 1, 60), [1]]))
    ty = np.sort(np.concatenate([[0], rng.uniform(0, 1, 45), [1]]))
    bx = 0.01 * np.cumsum(rng.standard_normal(len(tx)))  # log prices
    by = 0.5 * np.interp(ty, tx, bx) + 0.01 * np.cumsum(rng.standard_normal(len(ty)) * 0.3)
    d = hy_def(tx, bx, ty, by)
    g = {}
    try:
        import mfe.realized as m

        res = m.realized_hayashi_yoshida([np.exp(bx), np.exp(by)], [tx, ty])
        c = np.asarray(res.cov)
        g["mfe-toolbox realized_hayashi_yoshida"] = float(abs(c[0, 1] - d) / abs(d))
    except Exception as e:  # noqa: BLE001
        g["mfe-toolbox"] = f"ERROR {type(e).__name__}: {e}"
    try:
        from hfhd import hf

        h = hf.hayashi_yoshida
        import pandas as pd

        sx = pd.Series(bx, index=pd.to_datetime(np.round(tx * 1e6).astype(np.int64), unit="us").astype("datetime64[ns]"))
        sy = pd.Series(by, index=pd.to_datetime(np.round(ty * 1e6).astype(np.int64), unit="us").astype("datetime64[ns]"))
        c = np.asarray(h([sx, sy]))
        g["hfhd hayashi_yoshida"] = float(abs(c[0, 1] - d) / abs(d))
    except Exception as e:  # noqa: BLE001
        g["hfhd"] = f"ERROR {type(e).__name__}: {e}"
    report("499", "Hayashi-Yoshida covariance", "mfe-toolbox", g, "60 and 45 uniform ticks on [0,1], rel. gap to the O(nm) double sum")


def shifted_hy_contrast(tx, x, ty, y, theta):
    """Hoffmann-Rosenbaum-Yoshida 2013: |U_theta| / sqrt(sum dX^2 sum dY^2), Y's clock moved by -theta."""
    s = 0.0
    for i in range(1, len(tx)):
        for j in range(1, len(ty)):
            if tx[i - 1] < ty[j] - theta and ty[j - 1] - theta < tx[i]:
                s += (x[i] - x[i - 1]) * (y[j] - y[j - 1])
    return abs(s) / sqrt(np.sum(np.diff(x) ** 2) * np.sum(np.diff(y) ** 2))


@check
def lead_lag_hry():
    import pandas as pd
    from lead_lag import LeadLag

    rng = np.random.default_rng(3)
    # X leads Y by 20 ms: Y(t) = X(t - 0.020) + noise; integer-millisecond ticks
    T = 3000
    tx = np.unique(rng.integers(0, T, 400))
    ty = np.unique(rng.integers(0, T, 350))
    path = np.cumsum(rng.standard_normal(T + 100))
    x = path[tx + 20 - 0]  # X observes the path 20 ms ahead
    y = path[ty] + 0.0
    # lead-lag reads index.astype(int64) as ns: pandas 3 keeps the unit, so force ns
    idx = lambda t: pd.to_datetime(t.astype(np.int64), unit="ms").astype("datetime64[ns]")
    ll = LeadLag(pd.Series(x, index=idx(tx)), pd.Series(y, index=idx(ty)), max_lag=0.05)
    ll.run_inference()
    lags = ll.lag_range * ll.precision  # seconds
    got = np.asarray(ll.contrasts)
    out = {"lead_lag estimate (s)": float(ll.lead_lag)}
    for sign, name in ((1, "theta = +lag"), (-1, "theta = -lag")):
        d = np.array([shifted_hy_contrast(tx / 1e3, x, ty / 1e3, y, sign * l) for l in lags])
        out[f"max |contrast - definition|, {name}"] = float(np.max(np.abs(got - d)))
        out[f"argmax of definition, {name} (s)"] = float(lags[np.argmax(d)])
        if sign == 1:
            out["  rank-correlation of the two contrast curves"] = float(np.corrcoef(got, d)[0, 1])
            out["  lag of the largest gap (ms)"] = float(lags[np.argmax(np.abs(got - d))] * 1e3)
    report("499", "HRY time-shifted Hayashi-Yoshida lead-lag", "lead-lag", out, "400 and 350 ms ticks of one path, X ahead by 20 ms, lags -50..50 ms")


def cusum_def(x, mu0, k):
    """Page 1954, tabular form (Montgomery 6e, 9.1.2): C+_t = max(0, x - (mu0 + k) + C+_{t-1}); C-_t = max(0, (mu0 - k) - x + C-_{t-1})."""
    cp = cm = 0.0
    P, M = [], []
    for v in x:
        cp = max(0.0, v - (mu0 + k) + cp)
        cm = max(0.0, (mu0 - k) - v + cm)
        P.append(cp)
        M.append(cm)
    return np.array(P), np.array(M)


def page_hinkley_def(x, delta, alpha):
    """Page-Hinkley with a running mean and forgetting alpha (both directions), after Page 1954 / Hinkley 1971."""
    inc = dec = 0.0
    mn, mx = np.inf, -np.inf
    mean = 0.0
    ti, td = [], []
    for n, v in enumerate(x, 1):
        mean += (v - mean) / n
        inc = alpha * inc + (v - mean) - delta
        dec = alpha * dec + (v - mean) + delta
        mn, mx = min(mn, inc), max(mx, dec)
        ti.append(inc - mn)
        td.append(mx - dec)
    return np.array(ti), np.array(td)


@check
def cusum_candidates():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.standard_normal(150), 1.0 + rng.standard_normal(150)])  # mu0=0, sigma=1, shift 1 sigma at 150
    k = 0.5
    cp, cm = cusum_def(x, 0.0, k)
    # (1) pyspc: fixed target, k = std/2 (the tabular CUSUM itself); its C- is stored negated
    from pyspc import cusum as pcusum

    (vals, *_rest) = pcusum(target=0.0, std=1.0).plot(x, 1)
    g = {"C+": float(np.max(np.abs(np.array(vals[0]) - cp))), "C- (sign flipped)": float(np.max(np.abs(-np.array(vals[1]) - cm)))}
    report("508", "tabular two-sided CUSUM, fixed target", "pyspc", g, "300 N(0,1) with a 1-sigma shift at 150, k=0.5; k is fixed at std/2 by pyspc")
    # (2) river PageHinkley: running mean, both sides; read the private sums to compare the statistic
    from river import drift

    ph = drift.PageHinkley(min_instances=30, delta=0.005, threshold=1e9, alpha=1.0, mode="both")
    ti, td = page_hinkley_def(x, 0.005, 1.0)
    gi = gd = 0.0
    for t, v in enumerate(x):
        ph.update(float(v))
        gi = max(gi, abs((ph._sum_increase - ph._min_increase) - ti[t]))
        gd = max(gd, abs((ph._max_decrease - ph._sum_decrease) - td[t]))
    report("508", "Page-Hinkley (running-mean reference, both sides)", "river", {"up statistic": gi, "down statistic": gd}, "same stream, delta=0.005, alpha=1; private fields read")
    # (3) frouros CUSUM: running mean, upper side only, max(0, .)
    from frouros.detectors.concept_drift import CUSUM, CUSUMConfig

    det = CUSUM(config=CUSUMConfig(delta=0.005, lambda_=1e9, min_num_instances=30))
    mean, c, gap = 0.0, 0.0, 0.0
    for n, v in enumerate(x, 1):
        det.update(value=float(v))
        mean += (v - mean) / n
        c = max(0.0, c + v - mean - 0.005)
        gap = max(gap, abs(det.sum_ - c))
    report("508", "one-sided CUSUM, running-mean reference", "frouros", {"upper sum": gap}, "same stream, delta=0.005")
    # (4) the identity that ties the forms: with a fixed reference, C+_t = S_t - min_{s<=t} S_s
    S = np.cumsum(x - k)
    ident = np.max(np.abs(cp - (S - np.minimum.accumulate(np.minimum(S, 0.0)))))
    print(f"[508] identity C+ = S_t - min(0, min_s S_s), fixed reference: max gap {ident:.3e} (self-check of the definition)")


@check
def expectile_candidates():
    from scipy import optimize, stats

    rng = np.random.default_rng(5)
    n, tau = 400, 0.3
    X = rng.standard_normal((n, 3))
    y = 1.0 + X @ np.array([1.0, -2.0, 0.5]) + rng.standard_t(4, n)
    A = np.column_stack([np.ones(n), X])

    def loss(b):  # Newey & Powell 1987 asymmetric least squares
        e = y - A @ b
        return np.sum(np.where(e < 0, 1 - tau, tau) * e**2)

    ref = optimize.minimize(loss, np.zeros(4), method="BFGS", options={"gtol": 1e-12}).x
    # tight reference: IRLS to the fixed point (the loss is strictly convex and C1)
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(200):
        w = np.where(y - A @ b < 0, 1 - tau, tau)
        b = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))
    print(f"[501] definition: BFGS vs IRLS fixed point, max |gap| {np.max(np.abs(ref - b)):.3e}")
    # (1) scipy.stats.expectile: location only (the intercept-only fit)
    e = stats.expectile(y, alpha=tau)
    bb = optimize.minimize(lambda m: np.sum(np.where(y - m[0] < 0, 1 - tau, tau) * (y - m[0]) ** 2), [0.0], tol=1e-14).x[0]
    report("501", "expectile, intercept only", "scipy", {"|stats.expectile - minimiser|": float(abs(e - bb))}, "400 rows, tau=0.3")
    # (2) pygam ExpectileGAM with linear terms (lam=0: no penalty)
    from pygam import ExpectileGAM, l

    gam = ExpectileGAM(l(0) + l(1) + l(2), expectile=tau, lam=0.0, max_iter=500, tol=1e-12).fit(X, y)
    coef = np.asarray(gam.coef_)  # [term coefficients..., intercept]
    got = np.concatenate([[coef[-1]], coef[:3]])
    report("501", "expectile regression (linear terms)", "pygam", {"max |coef - definition|": float(np.max(np.abs(got - b)))}, "400 rows, 3 features, t(4) noise, tau=0.3, lam=0")


def main(argv):
    names = argv or list(CHECKS)
    for n in names:
        try:
            CHECKS[n]()
        except ImportError as e:
            print(f"[{n}] SKIP (not installed: {e})")
        except Exception as e:  # noqa: BLE001
            print(f"[{n}] ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
