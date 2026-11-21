"""Task 516: linear equality constraints `R beta = r` on a Gram-based fit.

The EW ridge solves from the Gram in mean form: with `m` the EW mean of x, `C`
its centered covariance, `c` the centered cross-covariance with y and `ybar`
the EW mean of y,

    A = (C + rho I)^-1        beta = A c        beta_0 = ybar - m'beta.

Constraints act on the slopes (the intercept is never constrained: it is the
free parameter that makes the centered system exact). The restricted
minimiser of  1/2 b'(C + rho I) b - c'b  subject to  R b = r  is, in closed
form (one extra q x q solve on the Gram, q = number of constraints):

    beta_c = beta - A R' (R A R')^-1 (R beta - r),        beta_0 = ybar - m'beta_c

A decaying prior or a zero-weight row changes nothing in this: the Gram is
whatever the decay and the weights made it, and the constraint is read at
solve time. A constraint set with `R A R'` singular (dependent rows, or a
constraint no slope can meet) is refused. State added: none that grows;
(R, r) is configuration, 8 q (k + 1) bytes if the state file carries it.
"""

from __future__ import annotations

import numpy as np

from .common import lam_of


class EwJoint:
    """EW mean-form moments of (x, y): W, m, C (centered, divisor W)."""

    def __init__(self, k: int):
        self.k = k
        self.W = 0.0
        self.m = np.zeros(k + 1)
        self.C = np.zeros((k + 1, k + 1))

    def update(self, x, y, lam: float, w: float) -> None:
        Wo = lam * self.W
        W2 = Wo + w
        self.W = W2
        if w == 0.0 or W2 == 0.0:
            return
        a, b = Wo / W2, w / W2
        d = np.append(np.asarray(x, float), y) - self.m
        self.C = a * self.C + a * b * np.outer(d, d)
        self.m = self.m + b * d

    def solve(self, rho: float = 0.0, R=None, r=None):
        """(beta_0, slopes) from the Gram; constrained where R is given."""
        k = self.k
        Cxx, c = self.C[:k, :k], self.C[:k, k]
        A = np.linalg.inv(Cxx + rho * np.eye(k))
        beta = A @ c
        if R is not None:
            R = np.atleast_2d(np.asarray(R, float))
            r = np.atleast_1d(np.asarray(r, float))
            RAR = R @ A @ R.T
            if np.linalg.matrix_rank(RAR) < R.shape[0]:
                raise ValueError("constraints dependent or unreachable: R A R' singular")
            beta = beta - A @ R.T @ np.linalg.solve(RAR, R @ beta - r)
        return self.m[k] - self.m[:k] @ beta, beta


def constraint_bytes(k: int, q: int) -> int:
    return 8 * q * (k + 1)


# ---------------------------------------------------------------------------
def kkt(Cxx, c, rho, R, r):
    """Second derivation: the KKT system of the equality-constrained quadratic."""
    k = Cxx.shape[0]
    q = R.shape[0]
    K = np.block([[Cxx + rho * np.eye(k), R.T], [R, np.zeros((q, q))]])
    return np.linalg.solve(K, np.concatenate([c, r]))[:k]


def report(seeds: int = 20) -> None:
    import statsmodels.api as sm
    from statsmodels.regression.recursive_ls import RecursiveLS

    from .common import rel, stream, table

    k = 4
    cons = {
        "sum = 1": (np.ones((1, k)), [1.0]),
        "b1 - 2 b2 = 0": (np.array([[1.0, -2.0, 0, 0]]), [0.0]),
        "sum = 1 and b3 = b4 + .3": (np.array([[1.0, 1, 1, 1], [0, 0, 1, -1]]), [1.0, 0.3]),
    }

    # (a) final fit with decay and weights vs GLM.fit_constrained (closed-form restricted WLS)
    rows = []
    for name, (R, r) in cons.items():
        for case, hl, zero in (("lam=1, unit w", np.inf, False), ("half_life 50, w in [.5,2], zero-weight rows", 50.0, True)):
            g = hold = 0.0
            for sd in range(seeds):
                n = 400
                x, y, dt, w = stream(sd, n, k)
                if np.isinf(hl):
                    w = np.ones(n)
                if zero:
                    w[2::6] = 0.0
                    w[0] = 0.0
                J = EwJoint(k)
                om = np.zeros(n)
                for i in range(n):
                    lam = lam_of(dt[i], hl)
                    om[:i] *= lam
                    om[i] = w[i]
                    J.update(x[i], y[i], lam, w[i])
                b0, b = J.solve(0.0, R, r)
                keep = om > 0
                X = sm.add_constant(x[keep], prepend=True)
                glm = sm.GLM(y[keep], X, family=sm.families.Gaussian(), var_weights=om[keep])
                res = glm.fit_constrained((np.hstack([np.zeros((len(r), 1)), R]), r))
                g = max(g, rel(np.concatenate([[b0], b]), res.params))
                hold = max(hold, float(np.max(np.abs(R @ b - r))))
            rows.append([name, case, g, hold])
    table("516a: final fit vs statsmodels GLM(Gaussian, var_weights).fit_constrained((R, q)) (largest relative gap over seeds) and the largest |R beta - r|",
          ["constraint", "case", "gap", "|R b - r|"], rows)

    # (b) with a ridge penalty: second derivation by the KKT system
    rows = []
    for rho in (0.0, 0.05, 2.0):
        g = 0.0
        for sd in range(seeds):
            x, y, dt, w = stream(sd, 300, k)
            J = EwJoint(k)
            for i in range(300):
                J.update(x[i], y[i], lam_of(dt[i], 40.0), w[i])
            R, r = cons["sum = 1 and b3 = b4 + .3"]
            _, b = J.solve(rho, R, r)
            g = max(g, rel(b, kkt(J.C[:k, :k], J.C[:k, k], rho, R, np.array(r))))
        rows.append([rho, g])
    table("516b: closed form vs the KKT block system, ridge rho (relative gap)", ["rho", "gap"], rows)

    # (c) row by row vs RecursiveLS(constraints=(R, q)), expanding, unit weights, ridge 0
    rows = []
    for name, (R, r) in cons.items():
        g = {}
        for sd in range(min(seeds, 8)):
            n = 120
            x, y, dt, w = stream(sd, n, k)
            X = sm.add_constant(x, prepend=True)
            Rf = np.hstack([np.zeros((len(r), 1)), R])
            # statsmodels 0.15 with a numpy endog and constraints indexes `endog[:, 1:]` of a 1-D
            # array (IndexError): the endog is passed with its q constraint columns, which it overwrites
            res = RecursiveLS(np.column_stack([y, np.zeros((n, len(r)))]), X, constraints=(Rf, np.asarray(r)[:, None])).fit()
            filt = res.recursive_coefficients.filtered  # (k+1, n): after row t
            J = EwJoint(k)
            for i in range(n):
                J.update(x[i], y[i], 1.0, 1.0)
                if i >= k + 3:
                    try:
                        b0, b = J.solve(0.0, R, r)
                    except np.linalg.LinAlgError:
                        continue
                    gap = float(np.max(np.abs(np.concatenate([[b0], b]) - filt[:, i])))
                    g[i] = max(g.get(i, 0.0), gap)
        early = max(v for i, v in g.items() if i < 12)
        late = max(v for i, v in g.items() if i >= 12)
        rows.append([name, early, late, max(g.values())])
    table("516c: row-by-row vs RecursiveLS(constraints=(R, q)).filtered, aligned on the same row (absolute gap; row indices 7-11 and 12-119; the filter is diffusely initialised and imposes constraints as fake observations)",
          ["constraint", "rows 7-11", "rows 12-119", "max"], rows)

    # (d) first row weight zero and zero-weight rows interleaved == gaps folded forward
    J1, J2 = EwJoint(k), EwJoint(k)
    x, y, dt, w = stream(3, 60, k)
    carry = 0.0
    R, r = cons["sum = 1"]
    J1.update(x[0], y[0], 0.5, 0.0)
    ok = bool(np.isfinite(J1.C).all())
    J1 = EwJoint(k)
    for i in range(60):
        if i % 5 == 3:
            J1.update(x[i], y[i], lam_of(dt[i], 12.0), 0.0)
            carry += dt[i]
            continue
        J1.update(x[i], y[i], lam_of(dt[i], 12.0), w[i])
        J2.update(x[i], y[i], lam_of(dt[i] + carry, 12.0), w[i])
        carry = 0.0
    table("516d: zero-weight rows (first-row finite; constrained slopes with the rows present vs the gap folded forward, relative gap)",
          ["first row w=0 finite", "gap"], [[ok, rel(J1.solve(0.0, R, r)[1], J2.solve(0.0, R, r)[1])]])

    # (e) a refused set
    try:
        EwJoint(k).solve(0.0, np.array([[1.0, 0, 0, 0], [2.0, 0, 0, 0]]), [1.0, 2.0])
        refused = False
    except (ValueError, np.linalg.LinAlgError):
        refused = True
    rows = [[kk, q, constraint_bytes(kk, q)] for kk in (4, 20, 100) for q in (1, 2, 5)]
    table(f"516e: dependent constraints refused: {refused}. Configuration bytes 8 q (k+1); no accumulator is added",
          ["k", "q", "bytes"], rows)
