"""Study 500: online GLMs (log link) by recursive IRLS, against statsmodels and glum.

Part 1: Poisson and Gamma, log link, no decay, no window.

The online estimator takes one Fisher-scoring (IRLS) step per row at the fit
before the row and keeps the sums

    G_t = G_{t-1} + w_t W_t x_t x_t'          W_t = (dmu/deta)^2 / V(mu) at beta_{t-1}
    h_t = h_{t-1} + w_t W_t z_t x_t           z_t = eta_t + (y_t - mu_t) / (dmu/deta)
    beta_t = (G_t + R I)^{-1} h_t

with eta_t = x_t' beta_{t-1}, mu_t = exp(eta_t). For the log link dmu/deta = mu,
so W = mu (Poisson, V = mu) and W = 1 (Gamma, V = mu^2). The prediction mu_t is
emitted before the update (out of sample). A zero-weight row changes nothing.

The batch fits minimise the same penalised objective in mean form,
    (1 / sum w) * sum w_i * halfdev_i(beta) + alpha / 2 * |beta|^2,  alpha = R / sum w,
which is the online objective sum w_i halfdev_i + R/2 |beta|^2 divided by sum w.

Data are generated from seeded numpy generators; nothing is read from disk.

Run: .venv/bin/python scripts/study_500.py --seeds 10 --out <dir>/results.json
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np

ETA_CLIP = 30.0  # |eta| bound before exp(); never reached in these runs (counted)


# ---------------------------------------------------------------- data


def make_data(family: str, k: int, n: int, seed: int, gamma_shape: float = 2.0):
    """Intercept + (k-1) N(0,1) features; eta has sd about 0.5 around 0.5."""
    rng = np.random.default_rng([500, seed, k, 0 if family == "poisson" else 1])
    X = np.empty((n, k))
    X[:, 0] = 1.0
    X[:, 1:] = rng.standard_normal((n, k - 1))
    beta = np.empty(k)
    beta[0] = 0.5
    beta[1:] = rng.normal(0.0, 0.5 / np.sqrt(k - 1), k - 1)
    mu = np.exp(X @ beta)
    if family == "poisson":
        y = rng.poisson(mu).astype(float)
    else:
        y = rng.gamma(gamma_shape, mu / gamma_shape)
    return X, y, beta


# ---------------------------------------------------------------- online


class OnlineGLM:
    """Recursive IRLS, log link. State: G (k x k), h (k), beta (k), ridge R."""

    def __init__(self, k: int, family: str, ridge: float, refit_rows: int = 0):
        self.family = family
        self.refit_rows = refit_rows
        self._buf: list[tuple[np.ndarray, float, float]] = []
        self.ridge = ridge
        self.G = np.zeros((k, k))
        self.h = np.zeros(k)
        self.beta = np.zeros(k)
        self.weight_sum = 0.0
        self.clipped = 0
        self._R = ridge * np.eye(k)

    def step(self, x: np.ndarray, y: float, w: float = 1.0) -> float:
        eta = float(x @ self.beta)
        if abs(eta) > ETA_CLIP:
            self.clipped += 1
            eta = float(np.clip(eta, -ETA_CLIP, ETA_CLIP))
        mu = np.exp(eta)
        pred = mu  # out of sample: before the update with y
        if w == 0.0:
            return pred  # advances nothing, divides by nothing
        W = mu if self.family == "poisson" else 1.0
        z = eta + (y - mu) / mu
        self.G += (w * W) * np.outer(x, x)
        self.h += (w * W * z) * x
        self.weight_sum += w
        # ridge > 0 keeps G + R I positive definite from the first row
        self.beta = np.linalg.solve(self.G + self._R, self.h)
        if self.refit_rows:
            self._buf.append((x.copy(), y, w))
            if len(self._buf) == self.refit_rows:
                self._refit()
        return pred

    def _refit(self) -> None:
        """Start-up remedy (the scan's glm_refit_rows): once the buffer holds
        refit_rows rows, fit them to convergence and rebuild G, h at that fit,
        so the early rows' weights are those of a converged fit, not of the
        poor fits that scored them. The buffer is then dropped: state stays
        bounded by refit_rows * (k + 2) floats."""
        Xb = np.array([b[0] for b in self._buf])
        yb = np.array([b[1] for b in self._buf])
        wb = np.array([b[2] for b in self._buf])
        beta = irls_ridge(Xb, yb, wb, self.family, self.ridge)
        eta = np.clip(Xb @ beta, -ETA_CLIP, ETA_CLIP)
        W = np.exp(eta) if self.family == "poisson" else np.ones_like(eta)
        self.G = (Xb * (wb * W)[:, None]).T @ Xb
        self.beta = beta
        self.h = (self.G + self._R) @ beta
        self._buf = []
        self.refit_rows = 0


# ---------------------------------------------------------------- batch


def irls_ridge(X, y, w, family, ridge, tol=1e-13, maxiter=200):
    """Penalised IRLS (sum form, ridge R) with step halving; used by the refit."""
    k = X.shape[1]
    beta = np.zeros(k)
    beta[0] = np.log(max(np.average(y, weights=w), 1e-12))

    def obj(b):
        eta = np.clip(X @ b, -ETA_CLIP, ETA_CLIP)
        mu = np.exp(eta)
        hd = (mu - y * eta) if family == "poisson" else (y / mu + eta)
        return float(w @ hd + 0.5 * ridge * b @ b)

    f = obj(beta)
    for _ in range(maxiter):
        eta = np.clip(X @ beta, -ETA_CLIP, ETA_CLIP)
        mu = np.exp(eta)
        W = mu if family == "poisson" else np.ones_like(mu)
        s = (y - mu) if family == "poisson" else (y - mu) / mu
        g = X.T @ (w * s) - ridge * beta
        H = (X * (w * W)[:, None]).T @ X + ridge * np.eye(k)
        d = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-10:
            fn = obj(beta + t * d)
            if fn <= f:
                break
            t *= 0.5
        beta = beta + t * d
        if abs(f - fn) <= tol * (1 + abs(f)) and np.abs(t * d).max() < 1e-12:
            f = fn
            break
        f = fn
    return beta


def sm_family(family: str):
    import statsmodels.api as sm

    link = sm.families.links.Log()
    return sm.families.Poisson(link) if family == "poisson" else sm.families.Gamma(link)


def batch_statsmodels(X, y, family, alpha):
    """Unpenalised IRLS (for bse) and the ridge fit with alpha matched.

    statsmodels' ridge path (fit_regularized, L1_wt=0) minimises
    -loglike(beta)/nobs + alpha/2 |beta|^2 by BFGS. For Gamma its loglike
    profiles the scale at each beta, which is not the penalised deviance; we
    pin scale=1 on the model's loglike and score so the objective is
    halfdev/nobs + alpha/2 |beta|^2, the one glum and the online fit use.
    """
    import statsmodels.api as sm

    m = sm.GLM(y, X, family=sm_family(family))
    irls = m.fit(method="IRLS", tol=1e-12, maxiter=200)
    mr = sm.GLM(y, X, family=sm_family(family))
    ll, sc = mr.loglike, mr.score
    mr.loglike = lambda p, scale=None: ll(p, scale=1.0)
    mr.score = lambda p, scale=None: sc(p, scale=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ridge = mr.fit_regularized(alpha=alpha, L1_wt=0.0, start_params=np.asarray(irls.params))
    beta_r = np.asarray(ridge.params)
    # residual gradient of the mean-form objective at the reported optimum
    grad = -np.asarray(sc(beta_r, scale=1.0)) / len(y) + alpha * beta_r
    return np.asarray(irls.params), np.asarray(irls.bse), beta_r, float(np.abs(grad).max())


def batch_glum(X, y, family, alpha):
    from glum import GeneralizedLinearRegressor

    g = GeneralizedLinearRegressor(
        family=family,
        link="log",
        alpha=alpha,
        l1_ratio=0.0,
        fit_intercept=False,  # intercept is column 0 and penalised like the rest
        gradient_tol=1e-10,
        max_iter=1000,
    )
    g.fit(X, y)
    return np.asarray(g.coef_)


# ---------------------------------------------------------------- self-checks


def check_zero_weight(family: str, k: int = 5, n: int = 300) -> None:
    """A zero-weight row (including the first) leaves the state bit-identical."""
    X, y, _ = make_data(family, k, n, seed=999)
    a, b = OnlineGLM(k, family, 1.0), OnlineGLM(k, family, 1.0)
    rng = np.random.default_rng(7)
    zero = rng.random(n) < 0.2
    zero[0] = True
    for i in range(n):
        b.step(X[i], y[i], 0.0 if zero[i] else 1.0)
        if not zero[i]:
            a.step(X[i], y[i], 1.0)
    assert np.array_equal(a.beta, b.beta) and np.array_equal(a.G, b.G)
    assert np.array_equal(a.h, b.h) and np.all(np.isfinite(b.beta))


def check_newton_form(family: str, k: int = 5, n: int = 500) -> float:
    """The sums form equals the increment form beta += (G+R)^{-1} w x s_t,
    s_t = (y - mu) (dmu/deta) / V(mu): (G_t+R) beta_{t-1} + w W x (z - eta) = h_t."""
    X, y, _ = make_data(family, k, n, seed=998)
    o = OnlineGLM(k, family, 1.0)
    beta = np.zeros(k)
    G = np.eye(k)  # G + R I with R = 1
    worst = 0.0
    for i in range(n):
        x = X[i]
        mu = np.exp(x @ beta)
        W = mu if family == "poisson" else 1.0
        s = (y[i] - mu) if family == "poisson" else (y[i] - mu) / mu
        G = G + W * np.outer(x, x)
        beta = beta + np.linalg.solve(G, x * s)
        o.step(x, y[i])
        worst = max(worst, float(np.abs(beta - o.beta).max()))
    return worst


# ---------------------------------------------------------------- study


def run_one(family, k, seed, ns, ridge, use_glum, refit_rows=0):
    n = max(ns)
    X, y, beta_true = make_data(family, k, n, seed)
    o = OnlineGLM(k, family, ridge, refit_rows)
    snaps = {}
    t0 = time.perf_counter()
    for i in range(n):
        o.step(X[i], y[i])
        if i + 1 in ns:
            snaps[i + 1] = o.beta.copy()
    t_online = time.perf_counter() - t0
    out = {
        "family": family,
        "k": k,
        "seed": seed,
        "online_s": t_online,
        "clipped": o.clipped,
        "at": {},
    }
    for m in ns:
        Xm, ym = X[:m], y[:m]
        alpha = ridge / m
        b_irls, bse, b_smr, smr_grad = batch_statsmodels(Xm, ym, family, alpha)
        bo = snaps[m]
        rec = {
            "sm_ridge_abs": float(np.abs(bo - b_smr).max()),
            "sm_ridge_se": float((np.abs(bo - b_smr) / bse).max()),
            "sm_irls_abs": float(np.abs(bo - b_irls).max()),
            "sm_irls_se": float((np.abs(bo - b_irls) / bse).max()),
            "sm_ridge_grad": smr_grad,
            "true_online_se": float((np.abs(bo - beta_true) / bse).max()),
            "true_batch_se": float((np.abs(b_irls - beta_true) / bse).max()),
            "true_online_abs": float(np.abs(bo - beta_true).max()),
            "true_batch_abs": float(np.abs(b_irls - beta_true).max()),
            "bse_max": float(bse.max()),
        }
        if use_glum:
            bg = batch_glum(Xm, ym, family, alpha)
            rec["glum_abs"] = float(np.abs(bo - bg).max())
            rec["glum_se"] = float((np.abs(bo - bg) / bse).max())
            rec["glum_vs_smridge_se"] = float((np.abs(bg - b_smr) / bse).max())
        out["at"][m] = rec
    return out


def summarise(results, ns):
    keys = list(results[0]["at"][ns[0]])
    table = {}
    for r in results:
        cfg = (r["family"], r["k"])
        for m in ns:
            for key in keys:
                table.setdefault(cfg, {}).setdefault(m, {}).setdefault(key, []).append(
                    r["at"][m][key]
                )
    lines = []
    for (fam, k), by_m in table.items():
        for m, vals in by_m.items():
            row = {kk: (float(np.mean(v)), float(np.max(v))) for kk, v in vals.items()}
            lines.append(
                {
                    "family": fam,
                    "k": k,
                    "n": m,
                    **{kk: {"mean": a, "max": b} for kk, (a, b) in row.items()},
                }
            )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--ns", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--families", nargs="+", default=["poisson", "gamma"])
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--refit-rows", type=int, default=0)
    ap.add_argument("--no-glum", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    use_glum = not a.no_glum
    if use_glum:
        try:
            import glum  # noqa: F401
        except ImportError:
            print("glum not importable; statsmodels only")
            use_glum = False
    for fam in a.families:
        check_zero_weight(fam)
        print(f"{fam}: zero-weight rows leave the state bit-identical: ok")
        print(f"{fam}: sums form vs increment form, max |dbeta| = {check_newton_form(fam):.2e}")
    results = []
    for fam in a.families:
        for k in a.ks:
            for s in range(a.seeds):
                t0 = time.perf_counter()
                r = run_one(fam, k, s, sorted(a.ns), a.ridge, use_glum, a.refit_rows)
                results.append(r)
                last = r["at"][max(a.ns)]
                print(
                    f"{fam} k={k} seed={s} {time.perf_counter() - t0:.1f}s "
                    f"sm_ridge_se={last['sm_ridge_se']:.3g} "
                    f"glum_se={last.get('glum_se', float('nan')):.3g}",
                    flush=True,
                )
    summary = summarise(results, sorted(a.ns))
    for row in summary:
        print(
            f"{row['family']:7s} k={row['k']:2d} n={row['n']:6d} "
            f"sm_ridge abs max {row['sm_ridge_abs']['max']:.2e} "
            f"SE mean/max {row['sm_ridge_se']['mean']:.3f}/{row['sm_ridge_se']['max']:.3f} | "
            + (
                f"glum abs max {row['glum_abs']['max']:.2e} "
                f"SE mean/max {row['glum_se']['mean']:.3f}/{row['glum_se']['max']:.3f} | "
                if use_glum
                else ""
            )
            + f"truth SE online/batch mean {row['true_online_se']['mean']:.2f}/"
            f"{row['true_batch_se']['mean']:.2f}"
        )
    if a.out:
        with open(a.out, "w") as f:
            json.dump(
                {"args": vars(a), "glum": use_glum, "results": results, "summary": summary},
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
