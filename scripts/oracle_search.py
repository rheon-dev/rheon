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
        return m.get("License-Expression") or (m.get("License") or "").splitlines()[0][:30] or "see PyPI"
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
