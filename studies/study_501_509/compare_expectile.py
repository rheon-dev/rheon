"""The expectile checks (task 501). Each returns (header, rows) for a table."""

from __future__ import annotations

import warnings

import expectile as E
import numpy as np
import statsmodels.api as sm
from scipy.stats import expectile as sp_expectile
from statsmodels.robust.norms import LeastSquares, MQuantileNorm

import data as D

TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def rel(a, b) -> float:
    """max |a - b| over max |b|: a coefficient vector's gap, relative."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def ulps(a: float, b: float) -> float:
    return abs(a - b) / np.spacing(abs(b))


def sm_iterated_wls(Z, y, v, tau, max_iter=100):
    """statsmodels WLS, iterated: sides from the last fit's residuals until
    no side changes (statsmodels' own pinv solve, not our normal equations)."""
    beta = sm.WLS(y, Z, weights=v).fit().params
    sides = (y - Z @ beta) > 0
    for it in range(1, max_iter + 1):
        a = np.where(sides, tau, 1 - tau)
        beta = sm.WLS(y, Z, weights=v * a).fit().params
        new = (y - Z @ beta) > 0
        if np.array_equal(new, sides):
            return beta, it
        sides = new
    return beta, -1


def sm_rlm(Z, y, v, tau):
    """RLM with MQuantileNorm(tau, LeastSquares()): Newey and Powell's
    asymmetric least squares. RLM takes no weights; the loss is homogeneous of
    degree 2, so rows scaled by sqrt(v) (the constant column too) give the
    weighted fit."""
    s = np.sqrt(v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sm.RLM(y * s, Z * s[:, None], M=MQuantileNorm(tau, LeastSquares())).fit(
            maxiter=200, tol=1e-14
        )
    return r.params


def stationarity(beta, Z, y, v, tau):
    """|sum v a r z| relative to sum |v a r z| (0 at the exact fit)."""
    r = y - Z @ beta
    g = (Z * (v * E.asym(r, tau) * r)[:, None]).sum(0)
    return float(np.max(np.abs(g)) / np.sum(np.abs(Z * (v * r)[:, None])))


# ---------------------------------------------------------------- scalar


def scalar_vs_scipy(seeds=20):
    """The exact (sorted) scalar expectile against scipy.stats.expectile."""
    rows = []
    for n in (5, 100, 10_000):
        for wkind, lam in (("ones", 1.0), ("random", 1.0), ("ones", 0.99), ("zeros", 0.995)):
            worst, worst_plain = 0.0, 0.0
            for s in range(seeds):
                _, y = D.regression(n, 1, s)
                w = D.decayed(D.weights(n, s, wkind), lam)
                unit = np.spacing(np.max(np.abs(y)))
                for tau in TAUS:
                    ref = float(sp_expectile(y, alpha=tau, weights=w))
                    ours = E.expectile_exact(y, tau, w)
                    plain = E.expectile_exact(y, tau, w, compensated=False)
                    worst = max(worst, abs(ours - ref) / unit)
                    worst_plain = max(worst_plain, abs(plain - ref) / unit)
            rows.append((n, wkind, lam, seeds * len(TAUS), worst, worst_plain))
    hdr = ("n", "weights", "lam", "cases", "max gap, fsum (ulps of max abs y)",
           "max gap, running sums (ulps of max abs y)")
    return hdr, rows


def scalar_online_vs_scipy(seeds=20):
    """The one-pass scalar expectile against scipy on every row, decayed."""
    rows = []
    for lam in (1.0, 0.999, 0.99):
        for n in (1_000, 20_000):
            for tau in (0.1, 0.5, 0.9):
                gaps = []
                for s in range(seeds):
                    _, y = D.regression(n, 1, s)
                    w = D.weights(n, s, "random")
                    m, _ = E.scalar_online(y, w, tau, lam)
                    ref = float(sp_expectile(y, alpha=tau, weights=D.decayed(w, lam)))
                    sd = float(np.std(y))
                    gaps.append(abs(m - ref) / sd)
                rows.append((lam, n, tau, float(np.mean(gaps)), float(np.max(gaps))))
    return ("lam", "n", "tau", "mean abs gap / sd", "max abs gap / sd"), rows


# ------------------------------------------------------------ regression


def refit_vs_oracles(seeds=10):
    """The exact refit (IRLS, our normal equations) against the WLS longhand's
    stationarity, statsmodels WLS iterated, and statsmodels RLM."""
    rows = []
    for lam in (1.0, 0.999):
        for wkind in ("ones", "zeros"):
            for tau in TAUS:
                g_sm, g_rlm, g_stat, it_ours, it_sm = 0.0, 0.0, 0.0, 0, 0
                for s in range(seeds):
                    Z, y = D.regression(2_000, 3, s)
                    v = D.decayed(D.weights(2_000, s, wkind), lam)
                    res = E.irls(Z, y, v, tau)
                    assert res.converged, res
                    b_sm, its = sm_iterated_wls(Z, y, v, tau)
                    b_rlm = sm_rlm(Z, y, v, tau)
                    g_sm = max(g_sm, rel(res.beta, b_sm))
                    g_rlm = max(g_rlm, rel(res.beta, b_rlm))
                    g_stat = max(g_stat, stationarity(res.beta, Z, y, v, tau))
                    it_ours = max(it_ours, res.passes)
                    it_sm = max(it_sm, its)
                rows.append((lam, wkind, tau, g_stat, g_sm, g_rlm, it_ours, it_sm))
    hdr = ("lam", "weights", "tau", "stationarity", "vs sm WLS iter", "vs sm RLM",
           "passes (ours)", "passes (sm)")
    return hdr, rows


def tau_half_is_wls(seeds=10):
    """At tau = 1/2 every side weighs 1/2: one pass is the decayed WLS."""
    worst = 0.0
    for s in range(seeds):
        for lam in (1.0, 0.99):
            Z, y = D.regression(1_000, 3, s)
            w = D.weights(1_000, s, "zeros")
            m, *_ = E.run_online(Z, y, w, 0.5, lam)
            b = E.wls(Z, y, D.decayed(w, lam))
            worst = max(worst, rel(m.beta, b))
    return worst


def online_vs_batch(seeds=10):
    """One pass (and one pass with the first rows refit) against the exact
    decayed expectile regression of every row: a statistical gap, in units of
    the batch fit's residual scale, as the RMS of the fitted-value gap."""
    rows = []
    for lam in (1.0, 0.999, 0.99):
        for n in (500, 5_000, 20_000):
            if lam == 0.99 and n > 5_000:
                continue
            for tau in (0.1, 0.5, 0.9):
                for refit in (0, 200):
                    gaps, yard = [], []
                    for s in range(seeds):
                        Z, y = D.regression(n, 3, s)
                        w = D.weights(n, s, "random")
                        m, *_ = E.run_online(Z, y, w, tau, lam, refit_rows=refit)
                        v = D.decayed(w, lam)
                        # sampling yardstick: sqrt(p / n_kish), the RMS fitted-value
                        # error of a fit of p coefficients, in residual-scale units
                        yard.append(np.sqrt(4 / (np.sum(v) ** 2 / np.sum(v * v))))
                        b = E.irls(Z, y, v, tau).beta
                        r = y - Z @ b
                        scale = np.sqrt(np.sum(v * r * r) / np.sum(v))
                        d = Z @ (m.beta - b)
                        gaps.append(np.sqrt(np.sum(v * d * d) / np.sum(v)) / scale)
                    rows.append((lam, n, tau, refit, float(np.mean(gaps)), float(np.max(gaps)),
                                 float(np.mean(yard))))
    hdr = ("lam", "n", "tau", "refit rows", "mean gap/scale", "max gap/scale",
           "yardstick sqrt(p/n_kish)")
    return hdr, rows


def window_vs_batch(seeds=4, n=1_500):
    """The window refit each row (warm-started IRLS) against statsmodels on
    exactly the window's rows, and the passes it took per row."""
    rows = []
    for window in (50, 200, 1_000):
        for lam in (1.0, 0.995):
            for tau in (0.1, 0.5, 0.9):
                gap, passes, fails, halv = 0.0, [], 0, 0
                for s in range(seeds):
                    Z, y = D.regression(n, 3, s, noise="t3")
                    w = D.weights(n, s, "zeros")
                    b, _, _, ps, f, h = E.run_window(Z, y, w, tau, window, lam)
                    lo = n - window
                    v = w[lo:] * lam ** np.arange(window - 1, -1, -1.0)
                    ref, _ = sm_iterated_wls(Z[lo:], y[lo:], v, tau)
                    gap = max(gap, rel(b, ref))
                    passes.append(ps[ps > 0])
                    fails += f
                    halv += h
                ps = np.concatenate(passes)
                rows.append((window, lam, tau, gap, float(ps.mean()), int(np.percentile(ps, 99)),
                             int(ps.max()), fails, halv))
    hdr = ("window", "lam", "tau", "vs sm (rel)", "mean passes", "p99 passes", "max passes",
           "cap hit", "halvings")
    return hdr, rows


def kkt(beta, Z, y, v, tau) -> float:
    """The stationarity condition's largest entry, relative to the data's own
    size (|sum v a r z| over sqrt(sum v y^2) sqrt(sum v |z|^2)): rounding
    level at the exact fit, whatever the residuals are."""
    r = y - Z @ beta
    g = np.abs((Z * (v * E.asym(r, tau) * r)[:, None]).sum(0)).max()
    size = np.sqrt(np.sum(v * y * y)) * np.sqrt(np.sum(v * (Z * Z).sum(1)))
    return float(g / size) if size > 0 else float(g)


def convergence_stress(cases=3_000):
    """Small, awkward problems: cold-start IRLS. Counts passes, caps, step
    halvings and how each refit ended, with its stationarity and its loss
    against the better of statsmodels' two fits."""
    rng = np.random.default_rng(12345)
    rows = []
    for name in ("n=p+1", "n=2p", "ties (integer y)", "duplicated rows", "outliers 1e6",
                 "tau 0.01/0.99"):
        mx, caps, halved, tot, why, worst_kkt, worst_loss = 0, 0, 0, 0, {}, 0.0, 0.0
        for c in range(cases):
            p = int(rng.integers(2, 6))
            n = {"n=p+1": p + 1, "n=2p": 2 * p}.get(name, int(rng.integers(2 * p, 8 * p)))
            Z = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
            y = Z @ rng.normal(size=p) + rng.standard_t(2, n)
            tau = float(rng.uniform(0.05, 0.95))
            if name == "ties (integer y)":
                Z[:, 1:] = np.round(Z[:, 1:])
                y = np.round(y)
            if name == "duplicated rows":
                idx = rng.integers(0, n // 2 + 1, n)
                Z, y = Z[idx], y[idx]
            if name == "outliers 1e6":
                y[rng.random(n) < 0.2] *= 1e6
            if name == "tau 0.01/0.99":
                tau = 0.01 if c % 2 else 0.99
            v = rng.uniform(0.0, 2.0, n)
            if np.linalg.matrix_rank((Z * v[:, None]).T @ Z) < p:
                continue
            res = E.irls(Z, y, v, tau)
            tot += 1
            mx = max(mx, res.passes)
            caps += not res.converged
            halved += res.halvings > 0
            why[res.reason] = why.get(res.reason, 0) + 1
            worst_kkt = max(worst_kkt, kkt(res.beta, Z, y, v, tau))
            b_sm, _ = sm_iterated_wls(Z, y, v, tau)
            ref = min(E.loss(b_sm, Z, y, v, tau), E.loss(sm_rlm(Z, y, v, tau), Z, y, v, tau))
            size = max(float(np.sum(v * y * y)), 1e-300)
            worst_loss = max(worst_loss, (E.loss(res.beta, Z, y, v, tau) - ref) / size)
        ends = ", ".join(f"{k} {why[k]}" for k in sorted(why))
        rows.append((name, tot, mx, caps, halved, ends, worst_kkt, worst_loss))
    hdr = ("case", "problems", "max passes", "cap hit", "needed a halving", "ended by",
           "max stationarity", "max loss over statsmodels' (rel)")
    return hdr, rows


def warm_vs_cold(seeds=5, n=3_000, window=500):
    """Passes per row in the window: warm start (sides at the fit before the
    row) against a cold start (least squares first) on the same windows."""
    rows = []
    for tau in (0.1, 0.5, 0.9):
        warm, cold = [], []
        for s in range(seeds):
            Z, y = D.regression(n, 5, s)
            w = D.weights(n, s, "random")
            beta = None
            for t in range(window, n, 7):
                lo = t - window + 1
                v = w[lo : t + 1]
                rw = E.irls(Z[lo : t + 1], y[lo : t + 1], v, tau, beta)
                rc = E.irls(Z[lo : t + 1], y[lo : t + 1], v, tau, None)
                warm.append(rw.passes)
                cold.append(rc.passes)
                beta = rw.beta
        rows.append((tau, float(np.mean(warm)), int(np.max(warm)), float(np.mean(cold)),
                     int(np.max(cold))))
    return ("tau", "warm mean", "warm max", "cold mean", "cold max"), rows


def zero_weight(n=400):
    """Rule 9: a zero-weight row advances the clock and learns nothing."""
    Z, y = D.regression(n, 3, 7)
    w = D.weights(n, 7, "random")
    out = []
    # 1. Three zero-weight rows first: no NaN in the state, nothing emitted
    #    before the first fit, and the fit equal to the stream without them.
    Z0 = np.vstack([Z[:3], Z])
    y0 = np.concatenate([y[:3], y])
    w0 = np.concatenate([[0.0, 0.0, 0.0], w])
    for lam in (1.0, 0.99):
        a, pa, na, _ = E.run_online(Z0, y0, w0, 0.8, lam)
        b, pb, nb, _ = E.run_online(Z, y, w, 0.8, lam)
        out.append((f"zero-weight first rows, one pass, lam={lam}",
                    bool(np.all(np.isfinite(a.state()))),
                    bool(np.array_equal(a.beta, b.beta)),
                    bool(np.array_equal(pa[3:], pb, equal_nan=True))))
    # 2. Unguarded: the first row's 0/0 poisons the state for ever.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u, *_ = E.run_online(Z0, y0, w0, 0.8, 1.0, guard=False)
    out.append(("same, division unguarded", bool(np.all(np.isfinite(u.state()))),
                bool(np.all(np.isfinite(u.beta))) if u.beta is not None else False, False))
    # 3. Zero-weight rows mid-stream: the fit does not move across them, the
    #    raw weight (n_eff) decays.
    wz = w.copy()
    wz[100:110] = 0.0
    m = E.OnlineExpectile(4, 0.8, 0.99)
    fits, neffs = [], []
    for t in range(120):
        m.step(Z[t], y[t], wz[t])
        fits.append(None if m.beta is None else m.beta.copy())
        neffs.append(m.W)
    same = all(np.array_equal(fits[99], fits[t]) for t in range(100, 110))
    decays = all(neffs[t] == neffs[t - 1] * 0.99 for t in range(100, 110))
    out.append(("zero-weight rows mid-stream (fit fixed, n_eff decays)", True, same, decays))
    # 4. A window whose rows all weigh 0: no fit, no NaN.
    wz = w.copy()
    wz[:60] = 0.0
    b, pr, ne, *_ = E.run_window(Z[:80], y[:80], wz[:80], 0.8, 50)
    first = int(np.argmax(np.isfinite(pr)))
    out.append((f"window of 50, rows 0-59 weigh 0 (first prediction: row {first})",
                bool(b is None or np.all(np.isfinite(b))),
                bool(np.all(np.isnan(pr[:61]))), first == 64))
    # 5. Scalar: zero-weight first rows.
    m1, o1 = E.scalar_online(y0, w0, 0.8)
    m2, o2 = E.scalar_online(y, w, 0.8)
    out.append(("scalar, zero-weight first rows", bool(np.isfinite(m1)), m1 == m2,
                bool(np.array_equal(o1[3:], o2, equal_nan=True))))
    return ("case", "state finite", "fit as without them", "outputs as without them"), out


def one_pass_inner():
    """The row's own side, judged before and after its update, on a stream:
    how often they differ (the argument says never, but for r_post = 0)."""
    flips, zeros, total = 0, 0, 0
    for s in range(5):
        Z, y = D.regression(3_000, 3, s)
        w = D.weights(3_000, s, "random")
        for tau in (0.1, 0.9):
            m = E.OnlineExpectile(4, tau, 0.995)
            for t in range(3_000):
                pred, _ = m.step(Z[t], y[t], w[t])
                if np.isnan(pred) or m.beta is None:
                    continue
                r0 = y[t] - pred
                r1 = y[t] - Z[t] @ m.beta
                total += 1
                if r1 == 0.0:
                    zeros += 1
                elif (r0 > 0) != (r1 > 0):
                    flips += 1
    return total, flips, zeros
