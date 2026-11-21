"""Task 511: shrinkage covariance on an exponentially weighted covariance.

Beside the mean-form recursion `ew_cov` already keeps (W, Q, m, C):

    W' = lam W + w     Q' = lam^2 Q + w^2        (Kish n = W^2 / Q)
    a = lam W / W'     b = w / W'
    d = x - m          m' = m + b d              C' = a C + a b d d'

the Ledoit-Wolf shrinkage needs two more centered quantities, kept in the same
mean form: the vector T = E[|z|^2 z] and the scalar F = E[|z|^4], z = x - m.
Shifting the old group's deviations by u = b d (mean-zero old group, so the
cross terms with E z vanish):

    T' = a (T - tr(C) u - 2 C u - |u|^2 u) + b (1-b)^3 |d|^2 d
    F' = a (F + 4 u'Cu + |u|^4 - 4 T.u + 2 |u|^2 tr C) + b (1-b)^4 |d|^4

(T, F read before C is overwritten.) The identities are polynomial in the
weights, so the same step with weight -w retracts a row: that is how this
prototype windows (rheon windows by subtracting a snapshot; same quantity).

Estimators, with S = C (divisor W, the biased covariance), p features,
mu = tr S / p, n = Kish (or the count W where `n_rule="count"`):

    Shrunk:  (1-s) S + s mu I                            s given
    OAS:     alpha = |S|_F^2 / p^2;  s = min((alpha + mu^2) / ((n+1)(alpha - mu^2/p)), 1)
             (scikit-learn's, which omits the paper's 2/p factor); p = 1 -> 0
    LW:      delta = (|S|_F^2 - p mu^2) / p;  beta = (F - |S|_F^2) / (p n);
             s = min(beta, delta) / delta  (0 where beta = 0); p = 1 -> 0
"""

from __future__ import annotations

import numpy as np


class ShrinkAcc:
    def __init__(self, k: int, track_fourth: bool = True):
        self.k = k
        self.track = track_fourth
        self.W = 0.0
        self.Q = 0.0
        self.m = np.zeros(k)
        self.C = np.zeros((k, k))
        self.T = np.zeros(k)
        self.F = 0.0

    def update(self, x, lam: float, w: float) -> None:
        """Decay by lam, then fold row x at weight w (w may be negative to retract)."""
        Wo = lam * self.W
        W2 = Wo + w
        self.Q = lam * lam * self.Q + w * w
        self.W = W2
        if w == 0.0 or W2 == 0.0:
            return  # clock advanced, nothing learned (hard rule 9: no 0/0)
        a, b = Wo / W2, w / W2
        d = np.asarray(x, float) - self.m
        u = b * d
        if self.track:
            tr = np.trace(self.C)
            Cu = self.C @ u
            uu = u @ u
            dd = d @ d
            F_old = self.F + 4 * (u @ Cu) + uu * uu - 4 * (self.T @ u) + 2 * uu * tr
            T_old = self.T - tr * u - 2 * Cu - uu * u
            self.F = a * F_old + b * (1 - b) ** 4 * dd * dd
            self.T = a * T_old + b * (1 - b) ** 3 * dd * d
        self.C = a * self.C + a * b * np.outer(d, d)
        self.m = self.m + u

    def kish(self) -> float:
        return self.W * self.W / self.Q

    # estimators -------------------------------------------------------
    def _n(self, n_rule: str) -> float:
        return self.kish() if n_rule == "kish" else self.W

    def mu(self) -> float:
        return np.trace(self.C) / self.k

    def s_oas(self, n_rule: str = "kish") -> float:
        p = self.k
        if p == 1:
            return 0.0
        S = self.C
        alpha = float(np.mean(S**2))
        mu = self.mu()
        num = alpha + mu * mu
        den = (self._n(n_rule) + 1) * (alpha - mu * mu / p)
        return 1.0 if den == 0 else min(num / den, 1.0)

    def s_lw(self, n_rule: str = "kish") -> float:
        p = self.k
        if p == 1:
            return 0.0
        S = self.C
        mu = self.mu()
        fro = float(np.sum(S**2))
        delta = (fro - p * mu * mu) / p
        beta = (self.F - fro) / (p * self._n(n_rule))
        beta = min(beta, delta)
        return 0.0 if beta == 0 else beta / delta

    def shrunk(self, s: float) -> np.ndarray:
        return (1 - s) * self.C + s * self.mu() * np.eye(self.k)

    # state ------------------------------------------------------------
    @staticmethod
    def extra_bytes(k: int, kind: str) -> int:
        """Bytes added to ew_cov's state. oas / fixed: 0 (tr S, |S|_F^2 and Kish n
        are read from C, Q, W already kept; a fixed intensity is configuration,
        8 bytes). lw: T (k f64) and F (1 f64)."""
        return {"fixed": 0, "oas": 0, "lw": 8 * (k + 1)}[kind]


def window_run(x, w, dt, half_life, window, k):
    """Row window by retraction: the last `window` rows, weights as decayed."""
    from .common import lam_of
    acc = ShrinkAcc(k)
    ring = []  # [x, current weight]
    out = []
    for i in range(len(x)):
        lam = lam_of(dt[i], half_life)
        for r in ring:
            r[1] *= lam
        acc.update(x[i], lam, w[i])
        ring.append([x[i], w[i]])
        if len(ring) > window:
            xo, wo = ring.pop(0)
            # the retracted row's weight is its decayed weight; the decay is
            # already applied to the accumulator's mass, so lam = 1 here
            acc.update(xo, 1.0, -wo)
            # Q is a sum of squared weights: retracting subtracts wo^2 and the
            # step above added it, so Q must lose 2 wo^2 in total
            acc.Q -= 2 * wo * wo
        out.append(acc)
    return acc


def offline(x, omega, k):
    """Two-pass weighted moments: the oracle for the recursion itself."""
    W = omega.sum()
    m = omega @ x / W
    z = x - m
    C = (z * omega[:, None]).T @ z / W
    n2 = (z * z).sum(1)
    T = (omega * n2) @ z / W
    F = (omega * n2 * n2).sum() / W
    return W, (omega**2).sum(), m, C, T, F


# ---------------------------------------------------------------------------
def report(seeds: int = 20) -> None:
    from sklearn.covariance import OAS, LedoitWolf, ShrunkCovariance

    from .common import lam_of, rel, stream, table, ulps

    rows = []
    for k in (2, 5, 20):
        # (1) expanding, unit weights, no decay: exact against scikit-learn
        g = {"oas": 0.0, "lw": 0.0, "shr": 0.0, "Cs": 0.0, "s_oas": 0.0, "s_lw": 0.0}
        gw = dict(g)
        for sd in range(seeds):
            n = 300
            x, _, dt, _ = stream(sd, n, k)
            acc = ShrinkAcc(k)
            for i in range(n):
                acc.update(x[i], 1.0, 1.0)
            o, l, s = OAS().fit(x), LedoitWolf().fit(x), ShrunkCovariance(shrinkage=0.3).fit(x)
            g["s_oas"] = max(g["s_oas"], abs(acc.s_oas() - o.shrinkage_))
            g["s_lw"] = max(g["s_lw"], abs(acc.s_lw() - l.shrinkage_))
            g["oas"] = max(g["oas"], rel(acc.shrunk(acc.s_oas()), o.covariance_))
            g["lw"] = max(g["lw"], rel(acc.shrunk(acc.s_lw()), l.covariance_))
            g["shr"] = max(g["shr"], rel(acc.shrunk(0.3), s.covariance_))
            # (2) a row window (retraction), unit weights
            wn = 60
            ac2 = window_run(x, np.ones(n), np.ones(n), np.inf, wn, k)
            xw = x[-wn:]
            o, l = OAS().fit(xw), LedoitWolf().fit(xw)
            gw["s_oas"] = max(gw["s_oas"], abs(ac2.s_oas() - o.shrinkage_))
            gw["s_lw"] = max(gw["s_lw"], abs(ac2.s_lw() - l.shrinkage_))
            gw["oas"] = max(gw["oas"], rel(ac2.shrunk(ac2.s_oas()), o.covariance_))
            gw["lw"] = max(gw["lw"], rel(ac2.shrunk(ac2.s_lw()), l.covariance_))
        rows.append([k, "expanding, lam=1", g["s_oas"], g["s_lw"], g["oas"], g["lw"], g["shr"]])
        rows.append([k, "window 60 (retract)", gw["s_oas"], gw["s_lw"], gw["oas"], gw["lw"], "-"])
    table("511a: online form vs scikit-learn (largest gap over seeds; s = intensity abs gap, covariance = relative gap)",
          ["k", "case", "OAS s", "LW s", "OAS cov", "LW cov", "Shrunk cov"], rows)

    # (3) decay and weights: the recursion against the two-pass weighted moments
    rows = []
    for k in (2, 5, 20):
        gm = np.zeros(4)
        for sd in range(seeds):
            n = 400
            x, _, dt, w = stream(sd, n, k, offset=1e3)
            w[::7] = 0.0
            w[0] = 0.0  # a zero-weight FIRST row
            acc = ShrinkAcc(k)
            om = np.zeros(n)
            t_lam = []
            for i in range(n):
                lam = lam_of(dt[i], 20.0)
                om[:i] *= lam
                om[i] = w[i]
                acc.update(x[i], lam, w[i])
            keep = om > 0
            W, Q, m, C, T, F = offline(x[keep], om[keep], k)
            sc_t = np.sqrt(F * np.trace(C))  # T's natural scale: Cauchy-Schwarz bound on |E |z|^2 z|
            gm = np.maximum(gm, [rel(acc.C, C), float(np.max(np.abs(acc.T - T)) / sc_t),
                                 rel(acc.F, F), rel(acc.kish(), W * W / Q)])
        rows.append([k, "half_life 20, w in [.5,2], zero weights, offset 1e3", *gm])
    table("511b: recursion vs two-pass weighted moments (largest gap over seeds; C, F, Kish relative; T relative to sqrt(F tr C))",
          ["k", "case", "C", "T", "F", "Kish n"], rows)

    # (4) frequency weights (integer) with n = W: exact against scikit-learn on repeated rows
    rows = []
    for k in (3, 8):
        g = 0.0
        for sd in range(seeds):
            x, _, dt, _ = stream(sd, 80, k)
            wi = np.random.default_rng(sd).integers(1, 4, 80)
            acc = ShrinkAcc(k)
            for i in range(80):
                acc.update(x[i], 1.0, float(wi[i]))
            xr = np.repeat(x, wi, axis=0)
            g = max(g, abs(acc.s_oas("count") - OAS().fit(xr).shrinkage_),
                    abs(acc.s_lw("count") - LedoitWolf().fit(xr).shrinkage_))
        rows.append([k, "integer weights, n = W", g])
    table("511c: frequency weights vs scikit-learn on repeated rows (intensity abs gap)", ["k", "case", "gap"], rows)

    # (5) zero-weight rows: advance the clock, learn nothing
    rows = []
    k = 4
    x, _, dt, w = stream(1, 200, k)
    acc1, acc2 = ShrinkAcc(k), ShrinkAcc(k)
    carry = 0.0
    for i in range(200):
        if i % 5 == 3:  # zero-weight row: 'acc1' sees it, 'acc2' folds its gap into the next row
            acc1.update(x[i], lam_of(dt[i], 15.0), 0.0)
            carry += dt[i]
            continue
        acc1.update(x[i], lam_of(dt[i], 15.0), w[i])
        acc2.update(x[i], lam_of(dt[i] + carry, 15.0), w[i])
        carry = 0.0
    rows.append(["w=0 rows interleaved vs gap folded forward", ulps(acc1.C, acc2.C), ulps(acc1.F, acc2.F), ulps(acc1.s_lw(), acc2.s_lw())])
    a0 = ShrinkAcc(k)
    a0.update(x[0], 0.5, 0.0)
    fin = bool(np.isfinite(a0.C).all() and np.isfinite(a0.F) and a0.W == 0.0)
    rows.append(["first row weight 0: state finite, W = 0", fin, "-", "-"])
    table("511d: zero-weight rows (ulps between the two; they differ by the rounding of lam^a lam^b against lam^(a+b))",
          ["case", "C", "F", "LW s"], rows)

    # (6) state formula
    rows = [[k, ShrinkAcc.extra_bytes(k, "oas"), ShrinkAcc.extra_bytes(k, "lw"),
             8 * k * (k + 1) // 2 + 8 * k + 8 * 3] for k in (2, 5, 20, 100)]
    table("511e: bytes added to ew_cov's state per accumulator (a fixed or OAS shrinkage adds none; LW adds 8(k+1)); last column: ew_cov's own C, m, W, Q, prior as the lower triangle packs them",
          ["k", "OAS/fixed", "LW", "ew_cov in memory (packed lower triangle)"], rows)
