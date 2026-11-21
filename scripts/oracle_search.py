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


def dcc_loglik_def(r, D, a, b, g=0.0):
    """Engle 2002 / Cappiello-Engle-Sheppard 2006 Gaussian DCC(ADCC) log-likelihood given the univariate vols D (T x N).

    eps = r/D; Q_bar = mean eps eps'; N_bar = mean n n', n = min(eps, 0);
    Q_t = (1-a-b) Q_bar - g N_bar + a eps_{t-1} eps_{t-1}' + g n_{t-1} n_{t-1}' + b Q_{t-1}; R_t = diag(Q)^-1/2 Q diag(Q)^-1/2.
    Returns (loglik over t>=1 of the full-H form, the list of R_t); Q_0 = Q_bar."""
    T, N = r.shape
    eps = r / D
    nn = np.minimum(eps, 0.0)
    Qb = eps.T @ eps / T
    Nb = nn.T @ nn / T
    Q = Qb.copy()
    ll = 0.0
    Rs = [np.corrcoef(np.eye(N))]
    for t in range(1, T):
        Q = (1 - a - b) * Qb - g * Nb + a * np.outer(eps[t - 1], eps[t - 1]) + g * np.outer(nn[t - 1], nn[t - 1]) + b * Q
        d = 1 / np.sqrt(np.diag(Q))
        R = Q * np.outer(d, d)
        H = R * np.outer(D[t], D[t])
        ll += -0.5 * (N * np.log(2 * np.pi) + np.linalg.slogdet(H)[1] + r[t] @ np.linalg.solve(H, r[t]))
        Rs.append(R)
    return ll, Rs


@check
def dcc_candidates():
    from mgarch import mgarch

    rng = np.random.default_rng(6)
    T, N = 400, 2
    L = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))
    r = rng.standard_normal((T, N)) @ L.T * 0.01
    D = np.sqrt(np.maximum(1e-8, pd_ewma(r)))
    m = mgarch("norm")
    m.rt = np.matrix(r - r.mean(0))
    m.T, m.N = T, N
    got = -m.mgarch_loglike((0.05, 0.9), D)  # its likelihood is -2x(sum of terms)? compare both scalings
    ll, _ = dcc_loglik_def(m.rt.A, D, 0.05, 0.9)
    report("493/514", "Gaussian DCC log-likelihood at given (a, b)", "mgarch", {"|-mgarch_loglike - definition|": float(abs(got - ll)), "|-mgarch_loglike/2 - definition|": float(abs(got / 2 - ll)), "definition": float(ll)}, "T=400, N=2, rho=0.5, vols from a fixed EWMA; mgarch computes log(D_t.sum()) and reshapes the returns for Q_bar")


def pd_ewma(r, lam=0.94):
    v = np.empty_like(r)
    v[0] = r.var(0)
    for t in range(1, len(r)):
        v[t] = lam * v[t - 1] + (1 - lam) * r[t - 1] ** 2
    return v


def linucb_def(X, arms, y, x, arm, alpha, lam):
    """Li, Chu, Langford, Schapire 2010 (disjoint): A = lam I + sum x x', theta = A^-1 b, score = theta'x + alpha sqrt(x'A^-1 x)."""
    m = arms == arm
    A = lam * np.eye(X.shape[1]) + X[m].T @ X[m]
    th = np.linalg.solve(A, X[m].T @ y[m])
    return float(th @ x + alpha * np.sqrt(x @ np.linalg.solve(A, x))), th, A


@check
def bandit_candidates():
    rng = np.random.default_rng(7)
    n, d = 150, 3
    X = rng.standard_normal((n, d))
    arms = rng.integers(0, 3, n)
    y = np.array([X[i] @ np.array([1.0, 0.5, -1.0]) * (1 + arms[i]) + 0.1 * rng.standard_normal() for i in range(n)])
    xq = rng.standard_normal(d)
    alpha, lam = 1.5, 1.0
    out = {}
    from mabwiser.mab import MAB, LearningPolicy
    import numpy as _np

    mab = MAB([0, 1, 2], LearningPolicy.LinUCB(alpha=alpha, l2_lambda=lam), seed=0)
    mab.fit(decisions=arms.tolist(), rewards=y.tolist(), contexts=X)
    ex = mab.predict_expectations(xq.reshape(1, -1))
    out["mabwiser LinUCB: max |expectation - definition|"] = max(abs(ex[a] - linucb_def(X, arms, y, xq, a, alpha, lam)[0]) for a in range(3))
    report("504", "LinUCB (disjoint) scores", "mabwiser", out, "150 rounds, 3 arms, d=3, alpha=1.5, lambda=1")
    # river: the policy's bonus is the posterior predictive sigma (which includes the 1/beta noise term)
    from river import bandit

    pol = bandit.LinUCBDisjoint(alpha=lam, beta=1.0, seed=0)
    for i in range(n):
        pol.update(int(arms[i]), {j: float(X[i, j]) for j in range(d)}, float(y[i]))
    ctx = {j: float(xq[j]) for j in range(d)}
    g = {}
    for a in range(3):
        dist = pol._bayes_lin_regs[a].predict_dist_one(ctx)
        _, th, A = linucb_def(X, arms, y, xq, a, 1.0, lam)
        # river score = mu + sigma, sigma^2 = 1/beta + x'A^-1 x  (alpha_li = 1 only if 1/beta -> 0)
        g[f"arm {a}: |mu - theta'x|"] = float(abs(dist.mu - th @ xq))
        g[f"arm {a}: |sigma^2 - x'A^-1 x| (noise term)"] = float(abs(dist.sigma**2 - xq @ np.linalg.solve(A, xq)))
    report("504", "LinUCB (disjoint) posterior of the arms", "river", g, "same rounds; river's bonus is sigma of the predictive, 1/beta included")


def _leaves(node):
    if node.__class__.__name__ == "Leaf":
        return 1
    return _leaves(node.l) + _leaves(node.r)


@check
def rrcf_candidate():
    """Guha, Mishra, Roy, Schrijvers 2016: CoDisp(x) = max over subtrees C containing x of |sibling(C)| / |C|, counted by hand."""
    import rrcf

    rng = np.random.default_rng(0)
    tree = rrcf.RCTree()
    pts = rng.standard_normal((60, 3))
    for i, p in enumerate(pts):
        tree.insert_point(p, index=i)
    gap = 0.0
    for i in range(60):
        node, best = tree.leaves[i], 0.0
        while node.u is not None:
            sib = node.u.r if node is node.u.l else node.u.l
            best = max(best, _leaves(sib) / _leaves(node))
            node = node.u
        gap = max(gap, abs(best - tree.codisp(i)))
    # forest scoring is the mean over trees (their anomaly score) - check that the mean-of-trees API reproduces the same recursion
    report("507", "robust random cut forest CoDisp (given the tree)", "rrcf", {"max |codisp - definition| over 60 points": float(gap)}, "one tree of 60 N(0,I3) points; the cut distribution itself is stochastic (not compared)")


def bocpd_def(x, hazard, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
    """Adams & MacKay 2007, Algorithm 1, Gaussian with a normal-gamma prior (Murphy 2007, 3.5 predictive); returns P(r_t | x_1..t) per step."""
    from scipy import stats

    mu, kap, al, be = np.array([mu0]), np.array([kappa0]), np.array([alpha0]), np.array([beta0])
    R = np.array([1.0])
    out = []
    for v in x:
        pred = stats.t.pdf(v, df=2 * al, loc=mu, scale=np.sqrt(be * (kap + 1) / (al * kap)))
        H = hazard(np.arange(len(R))) if callable(hazard) else hazard
        growth = R * pred * (1 - H)
        cp = np.sum(R * pred * H)
        R = np.concatenate([[cp], growth])
        R /= R.sum()
        out.append(R)
        be = np.concatenate([[beta0], be + kap * (v - mu) ** 2 / (2 * (kap + 1))])
        mu = np.concatenate([[mu0], (kap * mu + v) / (kap + 1)])
        kap = np.concatenate([[kappa0], kap + 1])
        al = np.concatenate([[alpha0], al + 0.5])
    return out


@check
def bocpd_candidates():
    from bocd import BayesianOnlineChangePointDetection as B, ConstantHazard, StudentT

    rng = np.random.default_rng(8)
    x = np.concatenate([rng.standard_normal(60), 4 + rng.standard_normal(40)])
    ref = bocpd_def(x, 1 / 100)
    det = B(ConstantHazard(100), StudentT(mu=0, kappa=1, alpha=1, beta=1))
    gap = 0.0
    for t, v in enumerate(x):
        det.update(v)
        gap = max(gap, float(np.max(np.abs(det.belief - ref[t]))))
    report("523", "BOCPD run-length posterior (constant hazard, Gaussian)", "bocd", {"max |belief - definition| over 100 steps": gap, "its beta update reads kappaT where Murphy has betaT": "see source distribution.py"}, "60 N(0,1) then 40 N(4,1), hazard 1/100, prior (0,1,1,1)")


@check
def bocpd_runlength_hazard():
    """bayesian_changepoint_detection.online_changepoint_detection takes hazard_func(arange(t+1)): a run-length-dependent hazard."""
    from functools import partial

    from bayesian_changepoint_detection import online_changepoint_detection as o

    rng = np.random.default_rng(8)
    x = np.concatenate([rng.standard_normal(60), 4 + rng.standard_normal(40)])
    haz = lambda r: 1 / (1 + np.exp(-(np.asarray(r, float) - 50) / 10))  # logistic in the run length
    R, _ = o.online_changepoint_detection(x, haz, o.StudentT(alpha=1, beta=1, kappa=1, mu=0))
    ref = bocpd_def(x, haz)
    gap = max(float(np.max(np.abs(R[: t + 2, t + 1] - ref[t]))) for t in range(len(x)))
    report("523", "BOCPD, run-length-dependent (logistic) hazard", "bayesian-changepoint-detection", {"max |R[:, t+1] - definition|": gap}, "same stream, logistic hazard centred at run length 50")


def parzen(x):
    return 1 - 6 * x**2 + 6 * x**3 if x <= 0.5 else 2 * (1 - x) ** 3


def rk_def(r, H, dof=False, form="h/(H+1)"):
    """Barndorff-Nielsen, Hansen, Lunde, Shephard 2008 eq. (2), Parzen kernel, k(h/(H+1)); optional n/(n-H) small-sample factor."""
    n = len(r)
    g = lambda h: float(np.sum(r[h:] * r[: n - h]))
    arg = (lambda h: h / (H + 1)) if form == "h/(H+1)" else (lambda h: (h - 1) / H)  # second: BNHLS 2008 k((h-1)/H)
    v = g(0) + 2 * sum(parzen(arg(h)) * g(h) for h in range(1, H + 1))
    return v * (n / (n - H) if dof else 1.0)


@check
def realized_library_candidate():
    import scipy.integrate as _si

    if not hasattr(_si, "simps"):  # realized-library 0.1.2 imports the removed scipy.integrate.simps
        _si.simps = _si.simpson
    if not hasattr(_si, "trapz"):
        _si.trapz = np.trapezoid
    from realized_library.estimators.variance import (
        bipower_variation as bpv,
        med_rv,
        multipower_variation as mpv,
        realized_kernel as rk,
        realized_variance as rv,
    )

    rng = np.random.default_rng(1)
    r = rng.standard_normal(500) * 1e-3 + 0.3 * np.concatenate([[0], rng.standard_normal(499)]) * 1e-4  # mild serial structure
    p = np.exp(np.concatenate([[0.0], np.cumsum(r)]))  # 501 prices -> 500 log returns
    lr = np.diff(np.log(p))
    n = len(lr)
    out = {}
    out["RV"] = rel(rv.compute(p), rv_def(lr))
    out["BPV (default scaling)"] = rel(bpv.compute(p), bpv_def(lr))
    out["BPV (correct_scaling_bias=False)"] = rel(bpv.compute(p, correct_scaling_bias=False), bpv_def(lr))
    out["MedRV"] = rel(med_rv.compute(p), medrv_def(lr))
    # tripower variation (BNS 2004): mu_{2/3}^-3 sum |r_i|^{2/3}|r_{i-1}|^{2/3}|r_{i-2}|^{2/3}
    mu23 = 2 ** (1 / 3) * gamma(5 / 6) / gamma(0.5)
    a = np.abs(lr) ** (2 / 3)
    tpv = float(mu23**-3 * np.sum(a[2:] * a[1:-1] * a[:-2]))
    for flag in (True, False):
        out[f"TPV (m=3, r=2, correct_scaling_bias={flag})"] = rel(mpv.compute(p, m=3, r=2, correct_scaling_bias=flag), tpv)
    for H in (3, 8):
        for dof in (False, True):
            out[f"Parzen RK H={H} dof_adjustment={dof}"] = rel(rk.compute(p, bandwidth=H, kernel="parzen", dof_adjustment=dof), rk_def(lr, H, dof))
    for H in (3, 8):
        out[f"Parzen RK H={H}, k((h-1)/H) form, no dof"] = rel(rk.compute(p, bandwidth=H, kernel="parzen", dof_adjustment=False), rk_def(lr, H, False, "(h-1)/H"))
    report("499", "RV, BPV, MedRV, TPV, realized kernel", "realized-library", out, "500 log returns; rel. gap to the definitions (BNS 2004, ADS 2012, BNHLS 2008)")


def dm_def(y, p1, p2, h, hln=True, one_sided=False):
    """Diebold & Mariano 1995 with Harvey, Leybourne & Newbold 1997: sqrt((T+1-2h+h(h-1)/T)/T) factor, Student t(T-1) reference.
    The one-sided p-value is for H1: P1 has the smaller loss (statistic negative)."""
    from scipy import stats

    d = (y - p1) ** 2 - (y - p2) ** 2
    T = len(d)
    dm = d - d.mean()
    gam = lambda k: float(np.sum(dm[k:] * dm[: T - k]) / T)
    V = (gam(0) + 2 * sum(gam(k) for k in range(1, h))) / T
    st = d.mean() / np.sqrt(V)
    if hln:
        st *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = stats.t.cdf(st, T - 1) if one_sided else 2 * stats.t.cdf(-abs(st), T - 1)
    return float(st), float(p)


@check
def dm_candidate():
    from dieboldmariano import dm_test

    rng = np.random.default_rng(9)
    T, h = 200, 3
    y = rng.standard_normal(T).cumsum() * 0.1
    p1 = y + rng.standard_normal(T) * 0.5
    p2 = y + rng.standard_normal(T) * 0.6
    out = {}
    for hln in (True, False):
        for one in (False, True):
            st, p = dm_def(y, p1, p2, h, hln, one)
            stat, pv = dm_test(y, p1, p2, h=h, harvey_correction=hln, one_sided=one)
            out[f"harvey={hln}, one_sided={one}: |stat gap| / |p gap|"] = f"{abs(stat - st):.2e} / {abs(pv - p):.2e}"
    report("521", "Diebold-Mariano (HLN, t reference, one-sided)", "dieboldmariano", out, "T=200, h=3, squared loss")


@check
def giacomini_white_candidate():
    """GW 2006 (tau = 1): Z_t = h_{t-1} d_t, h = (1, d_{t-1}); Wald S = T Zbar' Omega^-1 Zbar, Omega = mean Z Z', chi2(q)."""
    from scipy import stats
    from feval import gw

    rng = np.random.default_rng(10)
    T = 300
    e1 = rng.standard_normal(T) * 1.0
    e2 = rng.standard_normal(T) * 1.1
    L = np.column_stack([e1**2, e2**2])
    d = L[:, 0] - L[:, 1]
    H = np.column_stack([np.ones(T - 1), d[:-1]])
    Z = H * d[1:, None]
    zb = Z.mean(0)
    Om = Z.T @ Z / len(Z)
    n = len(Z)
    wald = float(n * zb @ np.linalg.solve(Om, zb))
    S, cval, pv = gw(L[1:], tau=1, H=H)
    Om_c = np.cov(Z.T, ddof=1)
    S1, _, _ = gw(L[1:], tau=1, H=H[:, :1])  # unconditional (q = 1): the DM case
    z1 = Z[:, 0]
    wald1 = float(n * z1.mean() ** 2 / np.mean(z1**2))
    q = wald / n
    uncentred_r2 = float(n * q / (1 + q))
    report(
        "521",
        "Giacomini-White conditional test (tau=1, instruments 1 and d_{t-1})",
        "feval",
        {"S (feval)": float(S), "Wald definition": wald, "T * uncentred R^2 form": uncentred_r2, "|S - Wald|": float(abs(S - wald)), "|S - R^2 form|": float(abs(S - uncentred_r2)), "|p - chi2 sf of S|": float(abs(pv - stats.chi2.sf(S, 2))), "Wald with the centred covariance": float(n * zb @ np.linalg.solve(Om_c, zb)), "q=1: |S - Wald|": float(abs(S1 - wald1)), "q=1: T*R^2 form": float(n * (wald1 / n) / (1 + wald1 / n))},
        "T=300, squared losses of two forecasters; feval installed from GitHub (not on PyPI), 0.0.2; its tau=2 with covar_style=sample raises TypeError",
    )


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
