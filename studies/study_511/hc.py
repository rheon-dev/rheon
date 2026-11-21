"""Task 512: HC0-HC3 standard errors from the out-of-sample residual.

The sandwich over z = (1, x), row weight w and step decay lam:

    B' = lam B + w z z'                          the bread
    M' = lam^2 M + (w c e z)(w c e z)'           the meat, one per type
    Cov = B^-1 M B^-1

`e = y - z'beta_{t-1}` is the out-of-sample residual (rule 2). Before the row
is folded the fit's leverage of the row against the decayed bread is

    g = w z' (lam B)^-1 z,       and the fold itself gives   h = g / (1 + g)

(`h` is the row's leverage in the fit that includes it), and the residual the
fit that includes the row leaves is, by the rank-one identity,

    e_in = e / (1 + g) = e (1 - h).

So the textbook types, read at the row's own time, are the one residual `e`
times a factor of `g` (squared residual scale `c^2`):

    HC0*: c^2 = (1-h)^2 = 1/(1+g)^2      e_in^2
    HC1*: HC0* times n/(n - p), n = Kish's W^2/Q, p = z's length
    HC2*: c^2 = (1-h)   = 1/(1+g)       e_in^2 / (1-h)
    HC3*: c^2 = 1                        (e_in/(1-h))^2 = e^2

HC3* is the out-of-sample HC0 that `emit_robust_se` computes today: that is
the sense in which it is "already near HC3". No default moves; the new types
are options.  `1+g` is the square of `error_inflation` when h is the
(decayed, Kish-scaled) leverage that output reads; here g is the exact
weighted one, which is what the identity needs.

State: the bread is shared; the meat is d x d per type kept (d = k + 1),
nothing in it grows with the stream.
"""

from __future__ import annotations

import numpy as np

from .common import lam_of

TYPES = ("hc0", "hc1", "hc2", "hc3")


class HCSandwich:
    def __init__(self, k: int, types=TYPES, warm: int | None = None):
        self.d = k + 1
        # rows folded into the bread before the meat starts: the fit's first out-of-sample
        # residuals (a fit on d rows) are far off and, at lam=1, never wash out (512c's last rows)
        self.warm = self.d if warm is None else warm
        self.types = tuple(types)
        self.B = np.zeros((self.d, self.d))
        self.b = np.zeros(self.d)
        self.W = 0.0
        self.Q = 0.0
        self.rows = 0
        self.M = {t: np.zeros((self.d, self.d)) for t in self._meats()}
        self.last = {}  # the last row's e, g (for the tests)

    def _meats(self):
        # hc1 shares hc0's meat (a scalar), so it adds no matrix
        return sorted({"hc0" if t == "hc1" else t for t in self.types})

    def update(self, x, y, lam: float, w: float):
        d = self.d
        z = np.concatenate([[1.0], np.asarray(x, float)])
        Bd = lam * self.B
        self.W = lam * self.W
        self.Q = lam * lam * self.Q
        self.b = lam * self.b
        for t in self.M:
            self.M[t] *= lam * lam
        e = g = np.nan
        if w > 0 and self.rows >= self.warm:
            try:
                bd = np.linalg.solve(Bd, self.b)
                v = np.linalg.solve(Bd, z)
                e = y - z @ bd
                g = w * z @ v
            except np.linalg.LinAlgError:
                pass
        if w > 0:
            self.B = Bd + w * np.outer(z, z)
            self.b = self.b + w * y * z
            self.W += w
            self.Q += w * w
            self.rows += 1
        else:
            self.B = Bd
        self.last = {"e": e, "g": g}
        if np.isfinite(e):
            c2 = {"hc0": 1.0 / (1 + g) ** 2, "hc2": 1.0 / (1 + g), "hc3": 1.0}
            for t in self.M:
                u = w * e * np.sqrt(c2[t]) * z
                self.M[t] += np.outer(u, u)

    def kish(self):
        return self.W * self.W / self.Q

    def cov(self, t: str):
        key = "hc0" if t == "hc1" else t
        Bi = np.linalg.inv(self.B)
        c = Bi @ self.M[key] @ Bi
        if t == "hc1":
            n = self.kish()
            c = c * n / (n - self.d)
        return c

    def beta(self):
        return np.linalg.solve(self.B, self.b)

    def state_bytes(self) -> int:
        """Bytes beyond the existing HC0 sandwich (bread, one meat) for these types."""
        return 8 * self.d * self.d * (len(self.M) - 1)


def extra_bytes(k: int, n_types: int, lags: int = 0) -> int:
    """Bytes added when `n_types` distinct meats (hc0/hc1 share one) are kept at once;
    lags L > 0 (HAC) multiply by (L+1) matrices + L scores per extra type."""
    d = k + 1
    return 8 * (n_types - 1) * (d * d * (lags + 1) + d * lags)


# ---------------------------------------------------------------------------
def report(seeds: int = 20) -> None:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import OLSInfluence
    from types import SimpleNamespace
    from statsmodels.stats import sandwich_covariance as sc

    from .common import rel, stream, table

    # (a) exact building blocks: e_in, h of the row against statsmodels on the prefix
    rows = []
    for name, hl, zero in (("lam=1, unit w", np.inf, False), ("half_life 30, w in [.5,2], zero-weight rows", 30.0, True)):
        ge = gh = gi = 0.0
        for sd in range(seeds):
            k, n = 3, 120
            x, y, dt, w = stream(sd, n, k)
            if np.isinf(hl):
                w = np.ones(n)
            if zero:
                w[5::9] = 0.0
            hc = HCSandwich(k)
            om = np.zeros(n)
            for i in range(n):
                lam = lam_of(dt[i], hl)
                om[:i] *= lam
                om[i] = w[i]
                hc.update(x[i], y[i], lam, w[i])
                if i in (30, 60, 119) and w[i] > 0 and np.isfinite(hc.last["e"]):
                    keep = om[: i + 1] > 0
                    X = sm.add_constant(x[: i + 1][keep], prepend=True)
                    sw = np.sqrt(om[: i + 1][keep])
                    # statsmodels' WLS influence is not the weighted leverage; the whitened OLS's is
                    r = sm.OLS(y[: i + 1][keep] * sw, X * sw[:, None]).fit()
                    e_in = r.resid[-1] / sw[-1]
                    h = OLSInfluence(r).hat_matrix_diag[-1]
                    e, g = hc.last["e"], hc.last["g"]
                    ge = max(ge, abs(e_in - e / (1 + g)) / max(abs(e_in), 1e-12))
                    gh = max(gh, abs(h - g / (1 + g)))
                    # out-of-sample residual of the fit on the prefix without the row
                    keep2 = keep.copy()
                    keep2[-1] = False
                    X2 = sm.add_constant(x[: i + 1][keep2], prepend=True)
                    r2 = sm.WLS(y[: i + 1][keep2], X2, weights=om[: i + 1][keep2] / (lam if False else 1)).fit()
                    z = np.concatenate([[1.0], x[i]])
                    gi = max(gi, abs(y[i] - z @ r2.params - e) / max(abs(e), 1e-12))
        rows.append([name, ge, gh, gi])
    table("512a: the row-time quantities against statsmodels on the prefix (e_in relative gap, h absolute gap, e out-of-sample relative gap)",
          ["case", "e_in = e/(1+g)", "h = g/(1+g)", "e vs WLS on prefix without the row"], rows)

    # (b) assembled sandwich vs statsmodels' HC machinery fed the same row-time residuals (exact)
    rows = []
    for name, hl in (("lam=1", np.inf), ("half_life 40", 40.0)):
        g = {t: 0.0 for t in ("hc0", "hc1")}
        for sd in range(seeds):
            k, n = 3, 300
            x, y, dt, w = stream(sd, n, k)
            if np.isinf(hl):
                w = np.ones(n)
            hc = HCSandwich(k, ("hc0", "hc1"))
            om = np.zeros(n)
            es = np.zeros(n)
            for i in range(n):
                lam = lam_of(dt[i], hl)
                om[:i] *= lam
                om[i] = w[i]
                hc.update(x[i], y[i], lam, w[i])
                if np.isfinite(hc.last["e"]):
                    es[i] = hc.last["e"] / (1 + hc.last["g"])  # HC0*: the in-sample residual at the row's time
            sw = np.sqrt(om)
            X = sm.add_constant(x, prepend=True) * sw[:, None]
            m = sm.OLS(y * sw, X)
            fake = SimpleNamespace(model=SimpleNamespace(pinv_wexog=np.linalg.pinv(X), exog=X),
                                   resid=sw * es, nobs=n, df_resid=n - 4, normalized_cov_params=np.linalg.inv(X.T @ X))
            # statsmodels' HC1 uses n/(n-p) with n the row count; Kish n replaces it here
            h0 = sc.cov_hc0(fake)
            g["hc0"] = max(g["hc0"], rel(hc.cov("hc0"), h0))
            nk = hc.kish()
            g["hc1"] = max(g["hc1"], rel(hc.cov("hc1"), h0 * nk / (nk - 4)))
        rows.append([name, g["hc0"], g["hc1"]])
    table("512b: assembled covariance vs statsmodels' sandwich (`cov_hc0` on whitened rows) fed the row-time residuals; HC1 by n/(n-p) at Kish n (relative gap)",
          ["case", "HC0*", "HC1*"], rows)

    # (c) statistical: against statsmodels' final-fit HC0..HC3, SE ratio over seeds
    for het in (False, True):
        rows = []
        for n in (100, 500, 2000):
            rat = {(a, b): [] for a in TYPES for b in TYPES}
            for sd in range(seeds):
                k = 3
                x, y, dt, w = stream(sd, n, k, hetero=het)
                hc = HCSandwich(k, warm=20)
                hu = HCSandwich(k)  # unprotected: the meat starts at the first row with a fit
                for i in range(n):
                    hc.update(x[i], y[i], 1.0, 1.0)
                    hu.update(x[i], y[i], 1.0, 1.0)
                X = sm.add_constant(x, prepend=True)
                ref = {t.lower(): np.sqrt(np.diag(sm.OLS(y, X).fit(cov_type=t.upper()).cov_params())) for t in TYPES}
                for a in TYPES:
                    se = np.sqrt(np.diag(hc.cov(a)))
                    for b in TYPES:
                        rat[(a, b)].append(se / ref[b])
                rat.setdefault("u", []).append(np.sqrt(np.diag(hu.cov("hc3"))) / ref["hc3"])
            for a in TYPES:
                v = np.array(rat[(a, a)])
                v0 = np.array(rat[("hc3", "hc0")])
                rows.append([n, a.upper() + "*", float(v.mean()), float(v.min()), float(v.max()), float(np.abs(v - 1).max())])
            v0 = np.array(rat[("hc3", "hc0")])
            rows.append([n, "existing oos HC0 (=HC3*) vs HC0", float(v0.mean()), float(v0.min()), float(v0.max()), float(np.abs(v0 - 1).max())])
            vu = np.array(rat["u"])
            rows.append([n, "HC3* with no warm-up (meat from row k+1)", float(vu.mean()), float(vu.min()), float(vu.max()), float(np.abs(vu - 1).max())])
        table(f"512c: SE ratio rheon type / statsmodels same type (final fit, unit weights, no decay; meat starts after 20 rows; k=3, {'heteroskedastic' if het else 'homoskedastic'}; over seeds and coefficients)",
              ["n", "type", "mean", "min", "max", "max |ratio-1|"], rows)

    rows = [[k, nt, extra_bytes(k, nt), extra_bytes(k, nt, lags=5)] for k in (3, 20, 100) for nt in (2, 3, 4)]
    table("512d: bytes added by keeping nt distinct meats at once (hc0/hc1 share one; one type alone adds nothing)",
          ["k", "nt", "bytes (HC)", "bytes (with L=5 HAC lags)"], rows)
