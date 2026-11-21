"""Study 492: the GARCH family against arch, one entry point, numpy only
(docs/records/STUDY-492-GARCH-FAMILY.md).

    study_492.py [garch|t|figarch|all] [--quick] [options]    # default: all

``garch``   part 1, Gaussian innovations: GARCH(1,1), GJR-GARCH(1,1), EGARCH(1,1), GARCH(2,2) online against
            arch's batch fits on the same rows (the base estimator is study 485's variance-targeted GARCH(1,1)).
``t``       part 2: Student-t and skew-t innovations (GARCH(1,1), GJR(1,1), GARCH(2,2) wired, not run).
``figarch`` part 3: FIGARCH(1,1) with a bounded truncation (a ring of the last M squared residuals).
``all``     the three in turn.  ``--quick`` runs a few seeds on short series, asserts the structural checks and
            loose bounds, and finishes in under two minutes; without it the family's default grid runs
            (``garch``: 50 seeds, ``t``: 50, ``figarch``: 10, all at 10,000 rows, which takes hours; give
            ``--seeds``/``--n`` or shard the grid with SHARD and NSHARDS to spread it over cores).

    study_492.py check [family] [--quick]        # gradient, filter and density checks (seconds)
    study_492.py one garch MODEL SEED [N]        # one case, printed (MODEL: garch11, gjr, egarch, garch22)
    study_492.py one t MODEL DIST SEED [N]       # MODEL: g11, gjr, g22; DIST: t, skewt
    study_492.py one figarch SEED [N]
    study_492.py grid FAMILY OUTDIR [SEEDS] [N]  # one JSON line per case, resumable, sharded by SHARD/NSHARDS
    study_492.py summary FAMILY OUTDIR           # the tables of the record, with the tolerance verdicts

Part 1 (Gaussian). Every model is filtered in one of two forms, with ``l = log sigma2``:

    GARCH(p, o, q) (arch's notation: p ARCH lags, o asymmetric lags, q GARCH lags)
        sigma2_t = omega + sum_i a_i e2_{t-i} + sum_j g_j 1[e_{t-j} < 0] e2_{t-j}
                         + sum_k b_k sigma2_{t-k}
        dsigma2_t = (1, e2 lags, 1[e<0] e2 lags, sigma2 lags) + sum_k b_k dsigma2_{t-k}
        dl_t = dsigma2_t / sigma2_t
    EGARCH(1, 1, 1), z = e / sigma, c = E|z| = sqrt(2 / pi)
        l_t  = omega + a (|z_{t-1}| - c) + g z_{t-1} + b l_{t-1}
        dl_t = (1, |z_{t-1}| - c, z_{t-1}, l_{t-1}) + (b - (a |z_{t-1}| + g z_{t-1}) / 2) dl_{t-1}

and each takes the same recursive quasi-maximum-likelihood step as the base's GARCH(1,1)
(scripts/study_485/garch.py), on the score of the Gaussian quasi-likelihood
``l_t = -1/2 (ln 2 pi + l_t + e_t^2 / sigma2_t)``:

    g_t = 1/2 (z_t^2 - 1) dl_t,   I_t = 1/2 dl_t dl_t'
    H_t = H_{t-1} + (I_t - H_{t-1}) / (t + t0)
    theta = Proj(theta + H_t^{-1} g_t / (t + t0))

``(z^2 - 1)`` is replaced by the base's Huber-clipped residual when ``robust_c > 0`` (running centring and
slope). ``Proj`` is the Euclidean projection (Dykstra's algorithm over half-spaces) onto arch's own constraint
set, with the base's ``1 - 1e-4`` stationarity margin and ``omega`` floor.

The start is the base's adopted one (study 485, request A12): the first 500 rows are buffered and emit
nothing; the core-held projected Newton optimizer (``batch_core.maximize``) fits them from a small grid, the
buffer is replayed at that fit to build ``H`` as an equal-weight mean, and steps run from row 501 at
``1 / (t + 500)``. arch is the oracle only.

Part 2 (Student-t and skew-t, Hansen 1994): the innovations ``z`` are standardised (mean 0, variance 1),
``e_t = sigma_t z_t``:

    Student-t    log f(z) = ln c - (nu + 1)/2 ln(1 + z^2/(nu - 2)),
                 c = G((nu+1)/2) / (sqrt(pi (nu - 2)) G(nu/2)),  nu > 2
    skew-t       log f(z) = ln b + ln c - (nu + 1)/2 ln(1 + m^2/(nu - 2)),
                 m = (b z + a)/(1 + sg lambda),  sg = sign(z + a/b),
                 a = 4 lambda c (nu - 2)/(nu - 1),  b^2 = 1 + 3 lambda^2 - a^2

Per-row log-likelihood ``l_t = log f(z_t) - 1/2 ln s_t``. Score in the GARCH parameters:
``g_t = 1/2 w_t dl_t``, Student-t ``w = (nu + 1) z^2/(nu - 2 + z^2) - 1``, skew-t
``w = (nu + 1) m b z/((1 + sg lambda)(nu - 2 + m^2)) - 1``. The shape score is d log f/d nu (closed form for
the t; a central difference of the closed-form density for the skew-t, nu and lambda); the online step takes
BHHH information (the running mean of the outer product of the whole score).

Part 3 (FIGARCH): the section below the part-2 code.
"""

# ruff: noqa: E501, E741, B023, B905, SIM108  (a study script: long derivation comments, `I` for information)
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from scipy.special import digamma

sys.path.insert(0, str(Path(__file__).resolve().parent / "study_485"))
import batch_core  # noqa: E402  (the base's core-held optimizer)

LN_2PI = math.log(2.0 * math.pi)
ABS_Z = math.sqrt(2.0 / math.pi)
MARGIN = 1e-4
OMEGA_FLOOR_FRAC = 1e-6
BACKCAST_ROWS = 75
WARM = 500
SEED0 = 492000


def backcast(r2: np.ndarray) -> float:
    w = 0.94 ** np.arange(min(BACKCAST_ROWS, r2.shape[0]))
    return float(np.sum(r2[: w.shape[0]] * w) / np.sum(w))


def project_halfspaces(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {A x >= b} by Dykstra's algorithm."""
    if np.all(A @ x >= b - 1e-15):
        return x
    y = x.copy()
    inc = np.zeros((A.shape[0], x.shape[0]))
    nrm = np.einsum("ij,ij->i", A, A)
    for _ in range(2000):
        prev = y.copy()
        for i in range(A.shape[0]):
            w = y + inc[i]
            v = float(A[i] @ w) - b[i]
            z = w - (v / nrm[i]) * A[i] if v < 0.0 else w
            inc[i] = w - z
            y = z
        if np.max(np.abs(y - prev)) < 1e-14:
            break
    # tidy rounding so the result is admissible
    for i in range(A.shape[0]):
        v = float(A[i] @ y) - b[i]
        if v < 0.0:
            y = y - (v / nrm[i]) * A[i] * (1.0 + 1e-12)
    return y


def project_halfspaces_metric(
    x: np.ndarray, A: np.ndarray, b: np.ndarray, hess: np.ndarray
) -> np.ndarray:
    """``argmin (y - x)' hess (y - x)`` over ``{A y >= b}``: with ``hess = L L'`` and ``u = L' y`` it is the
    Euclidean projection of ``L' x`` onto ``{A L^-T u >= b}``, which Dykstra's algorithm computes. A Euclidean
    projection of a Newton step is not an ascent direction of the objective when ``hess`` is far from the
    identity (the constraints bind on coordinates the likelihood barely separates)."""
    x = np.asarray(x, dtype=float)
    if np.all(A @ x >= b - 1e-15):
        return x
    ell = np.linalg.cholesky(hess)
    u = project_halfspaces(ell.T @ x, A @ np.linalg.inv(ell.T), b)
    y = np.linalg.solve(ell.T, u)
    return y


# --------------------------------------------------------------------- models


@dataclass
class GarchPOQ:
    p: int
    o: int
    q: int
    name: str = ""

    @property
    def k(self) -> int:
        return 1 + self.p + self.o + self.q

    def names(self) -> list[str]:
        return (
            ["omega"]
            + [f"alpha[{i + 1}]" for i in range(self.p)]
            + [f"gamma[{i + 1}]" for i in range(self.o)]
            + [f"beta[{i + 1}]" for i in range(self.q)]
        )

    def arch_kwargs(self) -> dict:
        return {"vol": "GARCH", "p": self.p, "o": self.o, "q": self.q}

    def constraints(self, floor: float) -> tuple[np.ndarray, np.ndarray]:
        k, p, o, q = self.k, self.p, self.o, self.q
        rows, rhs = [], []
        e = np.eye(k)
        rows.append(e[0])
        rhs.append(floor)
        for i in range(p):
            rows.append(e[1 + i])
            rhs.append(0.0)
        for j in range(o):  # alpha_j + gamma_j >= 0 (arch's)
            rows.append(e[1 + j] + e[1 + p + j])
            rhs.append(0.0)
        for kk in range(q):
            rows.append(e[1 + p + o + kk])
            rhs.append(0.0)
        s = np.zeros(k)
        s[1 : 1 + p] = -1.0
        s[1 + p : 1 + p + o] = -0.5
        s[1 + p + o :] = -1.0
        rows.append(s)
        rhs.append(-(1.0 - MARGIN))
        return np.array(rows), np.array(rhs)

    def project(self, theta: np.ndarray, scale: float) -> np.ndarray:
        A, b = self.constraints(OMEGA_FLOOR_FRAC * scale)
        return project_halfspaces(np.asarray(theta, dtype=float), A, b)

    def persistence(self, th: np.ndarray) -> float:
        p, o = self.p, self.o
        return float(
            np.sum(th[1 : 1 + p]) + 0.5 * np.sum(th[1 + p : 1 + p + o]) + np.sum(th[1 + p + o :])
        )

    # variance targeting: omega = v (1 - c . phi), phi the k - 1 free parameters
    def c_vec(self) -> np.ndarray:
        return np.concatenate([np.ones(self.p), 0.5 * np.ones(self.o), np.ones(self.q)])

    def project_free(self, phi: np.ndarray) -> np.ndarray:
        A, b = self.constraints(0.0)
        return project_halfspaces(np.asarray(phi, dtype=float), A[1:, 1:], b[1:])

    def grid(self, v: float) -> list[np.ndarray]:
        out = []
        for a in (0.02, 0.05, 0.1):
            for g in (0.0, 0.1) if self.o else (0.0,):
                for P in (0.8, 0.95, 0.98):
                    bsum = P - a - 0.5 * g
                    th = np.concatenate(
                        [
                            [v * (1.0 - P)],
                            np.full(self.p, a / self.p),
                            np.full(self.o, g / max(self.o, 1)),
                            np.full(self.q, bsum / self.q),
                        ]
                    )
                    out.append(th)
        return out

    # state: residual lags (max(p, o)), sigma2 lags (q), dsigma2 lags (q x k)
    def init(self, start: float) -> dict:
        m = max(self.p, self.o)
        return {
            "e": [math.sqrt(start)] * m,  # pre-sample e2 = start (arch's backcast)
            "a": [0.5 * start] * self.o,  # pre-sample asymmetric term: arch's 0.5 * backcast
            "s2": [start] * self.q,
            "d": [np.zeros(self.k) for _ in range(self.q)],
        }

    def predict(self, th: np.ndarray, st: dict) -> tuple[float, np.ndarray]:
        p, o, q = self.p, self.o, self.q
        e = st["e"]
        x = np.empty(self.k)
        x[0] = 1.0
        for i in range(p):
            x[1 + i] = e[i] * e[i]
        for j in range(o):
            x[1 + p + j] = st["a"][j]
        for kk in range(q):
            x[1 + p + o + kk] = st["s2"][kk]
        s2 = float(th @ x)
        d = x
        for kk in range(q):
            d = d + th[1 + p + o + kk] * st["d"][kk]
        return s2, d / s2 if s2 > 0.0 else d

    def push(self, st: dict, r: float, s2: float, dl: np.ndarray) -> None:
        if st["e"]:
            st["e"] = [r] + st["e"][:-1]
        if st["a"]:
            st["a"] = [r * r if r < 0.0 else 0.0] + st["a"][:-1]
        st["s2"] = [s2] + st["s2"][:-1]
        st["d"] = [dl * s2] + st["d"][:-1]

    def state_floats(self) -> int:
        k = self.k
        return k + k * (k + 1) // 2 + max(self.p, self.o) + self.o + self.q + self.q * k + 5


@dataclass
class Egarch11:
    name: str = "EGARCH(1,1)"
    k: int = 4

    def names(self) -> list[str]:
        return ["omega", "alpha[1]", "gamma[1]", "beta[1]"]

    def arch_kwargs(self) -> dict:
        return {"vol": "EGARCH", "p": 1, "o": 1, "q": 1}

    def project(self, theta: np.ndarray, scale: float) -> np.ndarray:
        th = np.asarray(theta, dtype=float).copy()
        th[3] = min(max(th[3], 0.0), 1.0 - MARGIN)  # arch: 0 <= beta < 1
        return th

    def persistence(self, th: np.ndarray) -> float:
        return float(th[3])

    def grid(self, v: float) -> list[np.ndarray]:
        lv = math.log(v)
        out = []
        for a in (0.05, 0.1, 0.2):
            for g in (-0.1, 0.0):
                for b in (0.8, 0.95, 0.98):
                    out.append(np.array([lv * (1.0 - b), a, g, b]))
        return out

    # state: z lag, l lag, dl lag (k)
    def init(self, start: float) -> dict:
        return {"z": None, "l": math.log(start), "d": np.zeros(4)}

    def predict(self, th: np.ndarray, st: dict) -> tuple[float, np.ndarray]:
        z = st["z"]
        if z is None:  # pre-sample news terms are zero
            az, zz, coef = 0.0, 0.0, th[3]
        else:
            az, zz = abs(z) - ABS_Z, z
            coef = th[3] - 0.5 * (th[1] * abs(z) + th[2] * z)
        lt = th[0] + th[1] * az + th[2] * zz + th[3] * st["l"]
        d = np.array([1.0, az, zz, st["l"]]) + coef * st["d"]
        return math.exp(min(lt, 700.0)), d

    def push(self, st: dict, r: float, s2: float, dl: np.ndarray) -> None:
        st["z"] = r / math.sqrt(s2)
        st["l"] = math.log(s2)
        st["d"] = dl

    def state_floats(self) -> int:
        k = self.k
        return k + k * (k + 1) // 2 + 1 + 1 + k + 5


MODELS = {
    "garch11": (GarchPOQ(1, 0, 1, "GARCH(1,1)"), np.array([0.05, 0.08, 0.90])),
    "gjr": (GarchPOQ(1, 1, 1, "GJR-GARCH(1,1)"), np.array([0.03, 0.05, 0.10, 0.87])),
    "egarch": (Egarch11(), np.array([0.0, 0.15, -0.08, 0.97])),
    "garch22": (GarchPOQ(2, 0, 2, "GARCH(2,2)"), np.array([0.03, 0.05, 0.05, 0.45, 0.42])),
}


def simulate(spec, theta: np.ndarray, n: int, seed: int, burn: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + burn)
    st = spec.init(1.0)
    out = np.empty(n + burn)
    for t in range(n + burn):
        s2, dl = spec.predict(theta, st)
        out[t] = math.sqrt(s2) * z[t]
        spec.push(st, out[t], s2, dl)
    return out[burn:]


def objective(spec, r: np.ndarray, start: float):  # noqa: ANN201
    """The Gaussian log-likelihood over ``r`` and its score and information."""

    def f(th: np.ndarray, grad: bool) -> tuple[float, np.ndarray, np.ndarray]:
        st = spec.init(start)
        ll = 0.0
        D = np.empty((r.shape[0], spec.k))
        res = np.empty(r.shape[0])
        for t in range(r.shape[0]):
            s2, dl = spec.predict(th, st)
            if not (s2 > 0.0 and math.isfinite(s2) and np.all(np.isfinite(dl))):
                return -math.inf, np.zeros(spec.k), np.zeros((spec.k, spec.k))
            x = float(r[t])
            ll -= 0.5 * (LN_2PI + math.log(s2) + x * x / s2)
            D[t] = dl
            res[t] = x * x / s2 - 1.0
            spec.push(st, x, s2, dl)
        if not grad:
            return ll, np.zeros(spec.k), np.zeros((spec.k, spec.k))
        return ll, 0.5 * D.T @ res, 0.5 * D.T @ D

    return f


def filter_path(spec, th: np.ndarray, r: np.ndarray, start: float) -> np.ndarray:
    st = spec.init(start)
    out = np.empty(r.shape[0])
    for t in range(r.shape[0]):
        s2, dl = spec.predict(th, st)
        out[t] = s2
        spec.push(st, float(r[t]), s2, dl)
    return out


def huber_constants(c: float) -> tuple[float, float]:
    inside = math.erf(c / math.sqrt(2.0))
    tail = 2.0 * c * math.exp(-0.5 * c * c) / math.sqrt(2.0 * math.pi)
    return inside - tail + c * c * (1.0 - inside), inside - tail


# ------------------------------------------------------------------ estimator


@dataclass
class Online:
    spec: object
    robust_c: float = 0.0
    target: bool = (
        False  # omega = v (1 - persistence), v the running mean of r^2 (study 485's form)
    )
    warm: int = WARM
    t0: float = float(WARM)
    buffer: list = field(default_factory=list)
    theta: np.ndarray | None = None
    H: np.ndarray | None = None
    st: dict | None = None
    n_seen: int = 0
    n_steps: int = 0
    mean_r2: float = 0.0
    rob_kappa: float = 0.0
    rob_slope: float = 0.0
    preds: list = field(default_factory=list)
    fit: object = None
    projected: int = 0
    start_seconds: float = 0.0

    def update(self, r: float) -> None:
        if self.theta is None:
            self.buffer.append(r)
            self.preds.append(math.nan)  # silent: buffering for the batch start
            if len(self.buffer) == self.warm:
                self._start()
            return
        self._row(r, step=True)

    def _start(self) -> None:
        t = time.perf_counter()
        rows = np.asarray(self.buffer)
        b = backcast(rows * rows)
        v = float(np.mean(rows * rows))
        spec = self.spec
        if self.target:
            c = spec.c_vec()
            J = np.vstack([-v * c, np.eye(spec.k - 1)])
            full = objective(spec, rows, b)

            def targeted(phi: np.ndarray, grad: bool):  # noqa: ANN202
                th = np.concatenate([[v * (1.0 - float(c @ phi))], phi])
                ll, g, I = full(th, grad)
                return ll, J.T @ g, J.T @ I @ J

            grid = [tuple(spec.project_free(g[1:])) for g in spec.grid(v)]
            self.fit = batch_core.maximize(targeted, spec.project_free, grid)
            phi = np.asarray(self.fit.theta, dtype=float)
            self.theta = np.concatenate([[v * (1.0 - float(c @ phi))], phi])
            self.H = np.zeros((spec.k - 1, spec.k - 1))
        else:
            grid = [tuple(spec.project(g, v)) for g in spec.grid(v)]
            self.fit = batch_core.maximize(
                objective(spec, rows, b), lambda th: spec.project(th, v), grid
            )
            self.theta = np.asarray(self.fit.theta, dtype=float).copy()
            self.H = np.zeros((spec.k, spec.k))
        self.st = spec.init(b)
        for x in rows:
            self._row(float(x), step=False)
        self.buffer = []
        self.start_seconds = time.perf_counter() - t

    def _row(self, r: float, step: bool) -> None:
        spec = self.spec
        s2, dl = spec.predict(self.theta, self.st)
        if step:
            self.preds.append(s2)  # out of sample: before this row's update
        r2 = r * r
        self.n_seen += 1
        self.mean_r2 += (r2 - self.mean_r2) / self.n_seen
        if self.target:
            vv = self.mean_r2
            dfree = dl[1:] - vv * spec.c_vec() * dl[0]
        else:
            dfree = dl
        info = 0.5 * np.outer(dfree, dfree)
        z2 = r2 / s2
        resid = z2 - 1.0
        if self.robust_c > 0.0:
            c2 = self.robust_c * self.robust_c
            clipped = min(z2, c2)
            self.rob_kappa += (clipped - self.rob_kappa) / self.n_seen
            self.rob_slope += ((z2 if z2 < c2 else 0.0) - self.rob_slope) / self.n_seen
            kappa, slope = self.rob_kappa, self.rob_slope
            resid = (clipped - kappa) / slope if slope > 0.0 else clipped - kappa
        if step:
            self.n_steps += 1
            gamma = 1.0 / (self.n_steps + self.t0)
            self.H += gamma * (info - self.H)
            move = gamma * np.linalg.solve(self.H, 0.5 * resid * dfree)
            if self.target:
                proposal = self.theta[1:] + move
                phi = spec.project_free(proposal)
                if not np.array_equal(phi, proposal):
                    self.projected += 1
                self.theta = np.concatenate([[vv * (1.0 - float(spec.c_vec() @ phi))], phi])
            else:
                proposal = self.theta + move
                new = spec.project(proposal, self.mean_r2)
                if not np.array_equal(new, proposal):
                    self.projected += 1
                self.theta = new
        else:
            self.H += (info - self.H) / self.n_seen
        spec.push(self.st, r, s2, dl)


# --------------------------------------------------------------------- oracle


def arch_fit(spec, r: np.ndarray):  # noqa: ANN201
    from arch import arch_model

    am = arch_model(r, mean="Zero", dist="normal", rescale=False, **spec.arch_kwargs())
    return am.fit(disp="off", options={"maxiter": 1000})


def compare_garch(model: str, seed: int, n: int = 10_000) -> dict:
    spec, truth = MODELS[model]
    r = simulate(spec, truth, n, seed)
    t = time.perf_counter()
    res = arch_fit(spec, r)
    arch_s = time.perf_counter() - t
    bp = np.asarray(res.params, dtype=float)
    se = np.asarray(res.std_err, dtype=float)
    path_b = np.asarray(res.conditional_volatility, dtype=float) ** 2
    ll_b = float(res.loglikelihood)
    out = {
        "model": model,
        "seed": seed,
        "n": n,
        "names": spec.names(),
        "truth": truth.tolist(),
        "batch": bp.tolist(),
        "se": se.tolist(),
        "arch_converged": int(res.convergence_flag) == 0,
        "arch_seconds": arch_s,
        "ll_batch": ll_b,
    }
    start = backcast(r * r)
    variants = [("plain", 0.0, False), ("huber2", 2.0, False)]
    if isinstance(spec, GarchPOQ):  # EGARCH has no closed-form target
        variants += [("target", 0.0, True), ("target_huber2", 2.0, True)]
    for tag, c, tg in variants:
        est = Online(spec, robust_c=c, target=tg)
        t = time.perf_counter()
        for x in r:
            est.update(float(x))
        secs = time.perf_counter() - t - est.start_seconds
        th = est.theta
        pred = np.asarray(est.preds)
        tail = slice(n - 1000, n)
        rel = np.abs(pred[tail] - path_b[tail]) / path_b[tail]
        half = slice(n // 2, n)
        relh = np.abs(pred[half] - path_b[half]) / path_b[half]
        ll_on = -0.5 * float(
            np.sum(LN_2PI + np.log(fp := filter_path(spec, th, r, start)) + r * r / fp)
        )
        gap = ((th - bp) / se).tolist()
        out[tag] = {
            "theta": th.tolist(),
            "gap_se": gap,
            "within2_all": bool(np.all(np.abs(gap) <= 2.0)),
            "within2_noomega": bool(np.all(np.abs(gap[1:]) <= 2.0)),
            "path_max_rel_last1000": float(np.max(rel)),
            "path_median_rel_last1000": float(np.median(rel)),
            "path_median_rel_second_half": float(np.median(relh)),
            "persistence_gap": spec.persistence(th) - spec.persistence(bp),
            "ll_online_params": ll_on,
            "start_stop": est.fit.stop,
            "start_iterations": est.fit.iterations,
            "start_seconds": est.start_seconds,
            "us_per_row": 1e6 * secs / (n - WARM),
            "projected_rows": est.projected,
            "silent_rows": int(np.sum(np.isnan(pred))),
        }
    return out


# ---------------------------------------------------------------------- check


def check_garch() -> None:
    """The analytic score against a central difference of the log-likelihood,
    and the fixed-parameter filter against arch's at arch's parameters."""
    for model, (spec, truth) in MODELS.items():
        r = simulate(spec, truth, 800, 7)
        start = backcast(r * r)
        f = objective(spec, r, start)
        _, g, _ = f(truth, True)
        num = np.empty(spec.k)
        for j in range(spec.k):
            h = 1e-6 * max(abs(truth[j]), 1e-2)
            e = np.zeros(spec.k)
            e[j] = h
            num[j] = (f(truth + e, False)[0] - f(truth - e, False)[0]) / (2 * h)
        relg = float(np.max(np.abs(g - num) / np.maximum(np.abs(num), 1e-3)))
        res = arch_fit(spec, simulate(spec, truth, 3000, 8))
        rr = simulate(spec, truth, 3000, 8)
        ours = filter_path(spec, np.asarray(res.params), rr, backcast(rr * rr))
        theirs = np.asarray(res.conditional_volatility) ** 2
        relp = float(np.max(np.abs(ours[500:] - theirs[500:]) / theirs[500:]))
        print(
            f"{model}: score vs central difference max rel {relg:.2e}; "
            f"filter vs arch after row 500 max rel {relp:.2e}; state {spec.state_floats()} floats"
        )
        assert relg < 1e-5, relg


def summary_garch_rows(rows: list[dict]) -> None:
    for model, (spec, _) in MODELS.items():
        rs = sorted((d for d in rows if d["model"] == model), key=lambda d: d["seed"])
        if not rs:
            continue
        print(
            f"\n{spec.name}: {len(rs)} seeds, n {rs[0]['n']}, arch converged "
            f"{sum(d['arch_converged'] for d in rs)}, arch fit median "
            f"{np.median([d['arch_seconds'] for d in rs]):.2f} s"
        )
        names = rs[0]["names"]
        for tag in [t for t in ("plain", "huber2", "target", "target_huber2") if t in rs[0]]:
            v = [d[tag] for d in rs]
            G = np.array([x["gap_se"] for x in v])
            worst_i = int(np.argmax(np.max(np.abs(G), axis=1)))
            wj = int(np.argmax(np.abs(G[worst_i])))
            out_ll = sum(x["ll_online_params"] > d["ll_batch"] + 1e-6 for x, d in zip(v, rs))
            print(
                f"  {tag}: within 2 SE all {sum(x['within2_all'] for x in v)}, "
                f"without omega {sum(x['within2_noomega'] for x in v)}; worst seed "
                f"{rs[worst_i]['seed']} {names[wj]} {G[worst_i, wj]:+.2f} SE; medians |gap| "
                + ", ".join(f"{nm} {np.median(np.abs(G[:, j])):.2f}" for j, nm in enumerate(names))
            )
            print(
                f"    sigma2 last 1000 rows: max rel gap, median over seeds "
                f"{np.median([x['path_max_rel_last1000'] for x in v]):.4f}, max "
                f"{np.max([x['path_max_rel_last1000'] for x in v]):.4f}; median rel gap, max over seeds "
                f"{np.max([x['path_median_rel_last1000'] for x in v]):.4f}; second half median, max "
                f"{np.max([x['path_median_rel_second_half'] for x in v]):.4f}"
            )
            print(
                f"    |persistence gap| max {np.max(np.abs([x['persistence_gap'] for x in v])):.4f}; "
                f"online out-scores batch on {out_ll}; start fits "
                f"{sum(x['start_stop'] == 'converged' for x in v)} converged, max iterations "
                f"{max(x['start_iterations'] for x in v)}, median start {np.median([x['start_seconds'] for x in v]):.2f} s; "
                f"projected rows total {sum(x['projected_rows'] for x in v)} on "
                f"{sum(x['projected_rows'] > 0 for x in v)} seeds; us/row median "
                f"{np.median([x['us_per_row'] for x in v]):.1f}; silent {v[0]['silent_rows']}"
            )
        print(
            f"  state: {spec.state_floats()} floats, {8 * spec.state_floats()} bytes (+ 500-row buffer, 4 KB, until the start)"
        )


# ===================================================================== part 2

T_SEED0 = 492100
NU_LO, NU_HI = 2.05, 300.0  # arch's bounds
LAM_LIM = 0.99
# ----------------------------------------------------------------- the densities


def consts(nu: float, lam: float) -> tuple[float, float, float]:
    c = math.exp(math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(
        math.pi * (nu - 2.0)
    )
    a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
    b = math.sqrt(1.0 + 3.0 * lam * lam - a * a)
    return a, b, c


def logf(z: np.ndarray, nu: float, lam: float) -> np.ndarray:
    """Log-density of the standardised skew-t; ``lam = 0`` is the Student-t."""
    a, b, c = consts(nu, lam)
    sg = np.where(z > -a / b, 1.0, -1.0)
    m = (b * z + a) / (1.0 + sg * lam)
    return math.log(b) + math.log(c) - 0.5 * (nu + 1.0) * np.log1p(m * m / (nu - 2.0))


def scores(dist: str, z: np.ndarray, nu: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """``(w, S)``: w the scale factor of the GARCH score, S (n, m) the shape scores."""
    if dist == "t":
        q = z * z
        w = (nu + 1.0) * q / (nu - 2.0 + q) - 1.0
        s_nu = (
            0.5 * (digamma((nu + 1.0) / 2.0) - digamma(nu / 2.0) - 1.0 / (nu - 2.0))
            - 0.5 * np.log1p(q / (nu - 2.0))
            + (nu + 1.0) * q / (2.0 * (nu - 2.0) * (nu - 2.0 + q))
        )
        return w, s_nu[:, None]
    a, b, c = consts(nu, lam)
    sg = np.where(z > -a / b, 1.0, -1.0)
    m = (b * z + a) / (1.0 + sg * lam)
    w = (nu + 1.0) * m * b * z / ((1.0 + sg * lam) * (nu - 2.0 + m * m)) - 1.0
    hn = 1e-5 * nu
    hl = 1e-6
    s_nu = (logf(z, nu + hn, lam) - logf(z, nu - hn, lam)) / (2.0 * hn)
    s_lam = (logf(z, nu, lam + hl) - logf(z, nu, lam - hl)) / (2.0 * hl)
    return w, np.stack([s_nu, s_lam], axis=1)


def shape_dim(dist: str) -> int:
    return 1 if dist == "t" else 2


def clip_shape(sh: np.ndarray) -> np.ndarray:
    out = np.asarray(sh, dtype=float).copy()
    out[0] = min(max(out[0], NU_LO), NU_HI)
    if out.shape[0] > 1:
        out[1] = min(max(out[1], -LAM_LIM), LAM_LIM)
    return out


# ------------------------------------------------------- the filter and objective


def sens_path(spec: GarchPOQ, th: np.ndarray, r: np.ndarray, start: float):  # noqa: ANN201
    st = spec.init(start)
    s2 = np.empty(r.shape[0])
    D = np.empty((r.shape[0], spec.k))
    for t in range(r.shape[0]):
        s, dl = spec.predict(th, st)
        if not (s > 0.0 and math.isfinite(s)):
            return None
        s2[t] = s
        D[t] = dl
        spec.push(st, float(r[t]), s, dl)
    return s2, D


def make_objective(spec: GarchPOQ, dist: str, r: np.ndarray, start: float):  # noqa: ANN201
    """Full-vector objective ``x = (theta, shape)``: log-likelihood, score, BHHH matrix."""
    k, m = spec.k, shape_dim(dist)

    def f(x: np.ndarray, grad: bool):  # noqa: ANN202
        th, sh = x[:k], x[k:]
        nu = float(sh[0])
        lam = float(sh[1]) if m > 1 else 0.0
        bad = (-math.inf, np.zeros(k + m), np.zeros((k + m, k + m)))
        if not (NU_LO <= nu <= NU_HI and abs(lam) < 1.0):
            return bad
        path = sens_path(spec, th, r, start)
        if path is None or not np.all(np.isfinite(path[1])):
            return bad
        s2, D = path
        z = r / np.sqrt(s2)
        ll = float(np.sum(logf(z, nu, lam) - 0.5 * np.log(s2)))
        if not (math.isfinite(ll)):
            return bad
        if not grad:
            return ll, bad[1], bad[2]
        w, S = scores(dist, z, nu, lam)
        P = np.hstack([0.5 * w[:, None] * D, S])
        return ll, P.sum(axis=0), P.T @ P

    return f


# --------------------------------------------------------------------- estimator


@dataclass
class OnlineT:
    spec: GarchPOQ
    dist: str  # "t" or "skewt"
    target: bool = False
    update_shape: bool = True
    metric: bool = False  # free form only: project the joint (theta, shape) step in the H metric
    warm: int = WARM
    t0: float = float(WARM)
    buffer: list = field(default_factory=list)
    theta: np.ndarray | None = None
    shape: np.ndarray | None = None
    H: np.ndarray | None = None
    st: dict | None = None
    n_seen: int = 0
    n_steps: int = 0
    mean_r2: float = 0.0
    preds: list = field(default_factory=list)
    fit: object = None
    projected: int = 0
    start_seconds: float = 0.0

    def update(self, r: float) -> None:
        if self.theta is None:
            self.buffer.append(r)
            self.preds.append(math.nan)
            if len(self.buffer) == self.warm:
                self._start()
            return
        self._row(r, step=True)

    def _start(self) -> None:
        t = time.perf_counter()
        spec, dist = self.spec, self.dist
        rows = np.asarray(self.buffer)
        b = backcast(rows * rows)
        v = float(np.mean(rows * rows))
        k, m = spec.k, shape_dim(dist)
        full = make_objective(spec, dist, rows, b)
        shapes = [(5.0, 0.0), (10.0, 0.0)] + ([(7.0, -0.2)] if m > 1 else [])
        if self.target:
            c = spec.c_vec()
            J = np.zeros((k + m, k - 1 + m))
            J[0, : k - 1] = -v * c
            J[1:k, : k - 1] = np.eye(k - 1)
            J[k:, k - 1 :] = np.eye(m)

            def expand(x: np.ndarray) -> np.ndarray:
                return np.concatenate([[v * (1.0 - float(c @ x[: k - 1]))], x[: k - 1], x[k - 1 :]])

            def targeted(x: np.ndarray, grad: bool):  # noqa: ANN202
                ll, g, I = full(expand(x), grad)
                return ll, J.T @ g, J.T @ I @ J

            def project(x: np.ndarray) -> np.ndarray:
                return np.concatenate([spec.project_free(x[: k - 1]), clip_shape(x[k - 1 :])])

            grid = [
                tuple(project(np.concatenate([g[1:], s[:m]]))) for g in spec.grid(v) for s in shapes
            ]
            self.fit = batch_core.maximize(targeted, project, grid)
            x = expand(np.asarray(self.fit.theta, dtype=float))
        else:

            def project(x: np.ndarray) -> np.ndarray:
                return np.concatenate([spec.project(x[:k], v), clip_shape(x[k:])])

            grid = [
                tuple(project(np.concatenate([g, s[:m]]))) for g in spec.grid(v) for s in shapes
            ]
            self.fit = batch_core.maximize(full, project, grid)
            x = np.asarray(self.fit.theta, dtype=float).copy()
        self.theta, self.shape = x[:k].copy(), x[k:].copy()
        n_free = (k - 1 if self.target else k) + (m if self.update_shape else 0)
        self.H = np.zeros((n_free, n_free))
        self.st = spec.init(b)
        for x_ in rows:
            self._row(float(x_), step=False)
        self.buffer = []
        self.start_seconds = time.perf_counter() - t

    def _metric_step(self, prop: np.ndarray, move: np.ndarray, nf: int) -> np.ndarray:
        """theta after the H-metric projection of the joint step. With the shape stepped, ``y_s`` is eliminated:
        the metric on theta is the Schur complement ``S = H11 - H12 H22^-1 H21`` and the shape moves by
        ``-H22^-1 H21 (theta' - prop_theta)`` on top of its own step (``_shape_shift``)."""
        hess = self.H + 1e-12 * np.eye(self.H.shape[0])
        a, b = self.spec.constraints(OMEGA_FLOOR_FRAC * self.mean_r2)
        self._shape_shift = np.zeros(self.H.shape[0] - nf)
        if self.H.shape[0] == nf:
            return project_halfspaces_metric(prop, a, b, hess)
        h11, h12, h22 = hess[:nf, :nf], hess[:nf, nf:], hess[nf:, nf:]
        w = np.linalg.solve(h22, h12.T)
        new = project_halfspaces_metric(prop, a, b, h11 - h12 @ w)
        self._shape_shift = -w @ (new - prop)
        return new

    def _row(self, r: float, step: bool) -> None:
        spec = self.spec
        s2, dl = spec.predict(self.theta, self.st)
        if step:
            self.preds.append(s2)
        self.n_seen += 1
        self.mean_r2 += (r * r - self.mean_r2) / self.n_seen
        nu = float(self.shape[0])
        lam = float(self.shape[1]) if self.shape.shape[0] > 1 else 0.0
        z = np.array([r / math.sqrt(s2)])
        w, S = scores(self.dist, z, nu, lam)
        if self.target:
            dfree = dl[1:] - self.mean_r2 * spec.c_vec() * dl[0]
        else:
            dfree = dl
        s = 0.5 * float(w[0]) * dfree
        if self.update_shape:
            s = np.concatenate([s, S[0]])
        if step:
            self.n_steps += 1
            gamma = 1.0 / (self.n_steps + self.t0)
            self.H += gamma * (np.outer(s, s) - self.H)
            move = gamma * np.linalg.solve(self.H + 1e-12 * np.eye(s.shape[0]), s)
            nf = dfree.shape[0]
            if self.target:
                prop = self.theta[1:] + move[:nf]
                phi = spec.project_free(prop)
                if not np.array_equal(phi, prop):
                    self.projected += 1
                self.theta = np.concatenate(
                    [[self.mean_r2 * (1.0 - float(spec.c_vec() @ phi))], phi]
                )
            else:
                prop = self.theta + move[:nf]
                new = spec.project(prop, self.mean_r2)
                if self.metric and not np.array_equal(new, prop):
                    new = self._metric_step(prop, move, nf)
                    move = np.concatenate([new - self.theta, move[nf:] + self._shape_shift])
                if not np.array_equal(new, prop):
                    self.projected += 1
                self.theta = new
            if self.update_shape:
                self.shape = clip_shape(self.shape + move[nf:])
        else:
            self.H += (np.outer(s, s) - self.H) / self.n_seen
        spec.push(self.st, r, s2, dl)

    def state_floats(self) -> int:
        k, m = self.spec.k, shape_dim(self.dist)
        nf = (k - 1 if self.target else k) + (m if self.update_shape else 0)
        sp = self.spec
        # theta, shape, H (nf x nf, symmetric), lags (e, asym, s2), q sensitivity lags, 3 scalars
        return k + m + nf * (nf + 1) // 2 + max(sp.p, sp.o) + sp.o + sp.q + sp.q * k + 3


# ------------------------------------------------------------------ data, oracle

T_MODELS = {
    "g11": (GarchPOQ(1, 0, 1, "GARCH(1,1)"), np.array([0.05, 0.08, 0.90]), ("t", "skewt")),
    "gjr": (GarchPOQ(1, 1, 1, "GJR-GARCH(1,1)"), np.array([0.03, 0.05, 0.10, 0.87]), ("t",)),
    "g22": (GarchPOQ(2, 0, 2, "GARCH(2,2)"), np.array([0.03, 0.05, 0.05, 0.45, 0.42]), ("t",)),
}
TRUE_SHAPE = {"t": np.array([6.0]), "skewt": np.array([6.0, -0.2])}


def draws(dist: str, n: int, seed: int) -> np.ndarray:
    nu = TRUE_SHAPE[dist][0]
    if dist == "t":
        rng = np.random.default_rng(seed)
        return rng.standard_t(nu, n) * math.sqrt((nu - 2.0) / nu)
    from arch.univariate.distribution import SkewStudent

    return np.asarray(SkewStudent(seed=np.random.default_rng(seed)).simulate(TRUE_SHAPE[dist])(n))


def simulate_t(spec: GarchPOQ, theta: np.ndarray, dist: str, n: int, seed: int, burn: int = 1000):
    z = draws(dist, n + burn, seed)
    st = spec.init(1.0)
    out = np.empty(n + burn)
    for t in range(n + burn):
        s2, dl = spec.predict(theta, st)
        out[t] = math.sqrt(s2) * z[t]
        spec.push(st, out[t], s2, dl)
    return out[burn:]


def arch_fit_t(spec: GarchPOQ, dist: str, r: np.ndarray):  # noqa: ANN201
    from arch import arch_model

    am = arch_model(r, mean="Zero", dist=dist, rescale=False, **spec.arch_kwargs())
    return am.fit(disp="off", options={"maxiter": 1000})


def loglik(spec: GarchPOQ, dist: str, x: np.ndarray, r: np.ndarray) -> float:
    return make_objective(spec, dist, r, backcast(r * r))(x, False)[0]


def compare_t(model: str, dist: str, seed: int, n: int = 10_000) -> dict:
    spec, truth, _ = T_MODELS[model]
    r = simulate_t(spec, truth, dist, n, seed)
    t = time.perf_counter()
    res = arch_fit_t(spec, dist, r)
    arch_s = time.perf_counter() - t
    bp = np.asarray(res.params, dtype=float)
    se = np.asarray(res.std_err, dtype=float)
    path_b = np.asarray(res.conditional_volatility, dtype=float) ** 2
    names = list(res.params.index)
    k = spec.k
    out = {
        "model": model,
        "dist": dist,
        "seed": seed,
        "n": n,
        "names": names,
        "truth": np.concatenate([truth, TRUE_SHAPE[dist]]).tolist(),
        "batch": bp.tolist(),
        "se": se.tolist(),
        "arch_converged": int(res.convergence_flag) == 0,
        "arch_seconds": arch_s,
        "ll_batch": float(res.loglikelihood),
    }
    for tg in (False, True):
        base = OnlineT(spec, dist, target=tg)
        started = None
        for i, x in enumerate(r):
            base.update(float(x))
            if i == WARM - 1:
                started = copy.deepcopy(base)  # the batch start is shared by both shape modes
                break
        for upd in (True, False):
            tag = ("target_" if tg else "plain_") + ("online" if upd else "fixed")
            est = copy.deepcopy(started)
            est.update_shape = upd
            n_free = (k - 1 if tg else k) + (shape_dim(dist) if upd else 0)
            est.H = np.zeros((n_free, n_free))
            est.st = spec.init(backcast(r[:WARM] ** 2))
            est.n_seen = 0
            est.mean_r2 = 0.0
            for x_ in r[:WARM]:  # replay the buffer at the start's fit to build H
                est._row(float(x_), step=False)
            est.preds = [math.nan] * WARM
            t = time.perf_counter()
            for x in r[WARM:]:
                est.update(float(x))
            secs = time.perf_counter() - t
            full = np.concatenate([est.theta, est.shape])
            pred = np.asarray(est.preds)
            tail = slice(n - 1000, n)
            rel = np.abs(pred[tail] - path_b[tail]) / path_b[tail]
            gap = ((full - bp) / se).tolist()
            out[tag] = {
                "x": full.tolist(),
                "gap_se": gap,
                "within2_all": bool(np.all(np.abs(gap) <= 2.0)),
                "within2_noomega": bool(np.all(np.abs(gap[1:]) <= 2.0)),
                "shape_gap_abs": (full[k:] - bp[k:]).tolist(),
                "shape_start": est.shape.tolist() if not upd else None,
                "path_max_rel_last1000": float(np.max(rel)),
                "path_median_rel_last1000": float(np.median(rel)),
                "persistence_gap": spec.persistence(full[:k]) - spec.persistence(bp[:k]),
                "ll_gap": loglik(spec, dist, full, r) - float(res.loglikelihood),
                "start_stop": est.fit.stop,
                "start_iterations": est.fit.iterations,
                "start_seconds": est.start_seconds,
                "us_per_row": 1e6 * secs / (n - WARM),
                "projected_rows": est.projected,
                "state_floats": est.state_floats(),
            }
    return out


# ------------------------------------------------------------------------ check


def check_t() -> None:
    from scipy import integrate

    spec, truth, _ = T_MODELS["g11"]
    for dist in ("t", "skewt"):
        r = simulate_t(spec, truth, dist, 800, 7)
        x0 = np.concatenate([truth, TRUE_SHAPE[dist]])
        f = make_objective(spec, dist, r, backcast(r * r))
        _, g, _ = f(x0, True)
        num = np.empty(x0.shape[0])
        for j in range(x0.shape[0]):
            h = 1e-5 * max(abs(x0[j]), 1e-2)
            e = np.zeros(x0.shape[0])
            e[j] = h
            num[j] = (f(x0 + e, False)[0] - f(x0 - e, False)[0]) / (2 * h)
        rel = np.abs(g - num) / np.maximum(np.abs(num), 1e-3)
        print(
            f"{dist}: score (theta, shape) vs central difference of the likelihood, max rel "
            f"{rel.max():.2e} (theta {rel[: spec.k].max():.2e}, shape {rel[spec.k :].max():.2e})"
        )
        assert rel[: spec.k].max() < 1e-5
        # likelihood at arch's own parameters (exact, with constants) against arch's
        res = arch_fit_t(spec, dist, r)
        ours = loglik(spec, dist, np.asarray(res.params), r)
        print(
            f"{dist}: log-likelihood at arch's parameters: ours - arch = {ours - res.loglikelihood:.2e}"
        )
        # density integrates to 1, mean 0, variance 1
        nu, lam = TRUE_SHAPE[dist][0], (TRUE_SHAPE[dist][1] if dist == "skewt" else 0.0)
        pdf = lambda z: float(np.exp(logf(np.array([z]), nu, lam))[0])  # noqa: E731
        i0 = integrate.quad(pdf, -np.inf, np.inf, limit=400)[0]
        i1 = integrate.quad(lambda z: z * pdf(z), -np.inf, np.inf, limit=400)[0]
        i2 = integrate.quad(lambda z: z * z * pdf(z), -np.inf, np.inf, limit=400)[0]
        print(f"{dist}: density integral {i0:.8f}, mean {i1:.2e}, variance {i2:.8f}")
    # expected information of the t: E w^2 = 2 nu/(nu + 3)
    rng = np.random.default_rng(1)
    for nu in (4.0, 6.0, 12.0):
        z = rng.standard_t(nu, 4_000_000) * math.sqrt((nu - 2.0) / nu)
        w, _ = scores("t", z, nu, 0.0)
        print(f"t nu={nu}: mean w^2 {np.mean(w * w):.4f} vs 2nu/(nu+3) {2 * nu / (nu + 3):.4f}")
    # the t's shape score closed form vs central difference of log f, and its mean (should be 0)
    z = rng.standard_t(6.0, 200_000) * math.sqrt(4.0 / 6.0)
    _, S = scores("t", z, 6.0, 0.0)
    h = 1e-5
    Sn = (logf(z, 6.0 + h, 0.0) - logf(z, 6.0 - h, 0.0)) / (2 * h)
    print(
        f"t: shape score closed form vs difference max abs {np.max(np.abs(S[:, 0] - Sn)):.2e}, mean {S.mean():.2e}"
    )
    zs = draws("skewt", 400_000, 3)
    _, S2 = scores("skewt", zs, 6.0, -0.2)
    print(
        f"skew-t: mean score (nu, lambda) at truth {S2.mean(axis=0)}, SE {S2.std(axis=0) / math.sqrt(len(zs))}"
    )


def summary_t_rows(rows: list[dict]) -> None:
    for model, (spec, _, dists) in T_MODELS.items():
        for dist in dists:
            rs = sorted(
                (d for d in rows if d["model"] == model and d["dist"] == dist),
                key=lambda d: d["seed"],
            )
            if not rs:
                continue
            names = rs[0]["names"]
            print(
                f"\n{spec.name} {dist}: {len(rs)} seeds, n {rs[0]['n']}, arch converged "
                f"{sum(d['arch_converged'] for d in rs)}, arch fit median "
                f"{np.median([d['arch_seconds'] for d in rs]):.2f} s"
            )
            B = np.array([d["batch"] for d in rs])
            print(
                "  arch mean "
                + ", ".join(f"{nm} {B[:, j].mean():.3f}" for j, nm in enumerate(names))
            )
            for tag in [t for t in rs[0] if t.startswith(("plain_", "target_"))]:
                v = [d[tag] for d in rs]
                G = np.array([x["gap_se"] for x in v])
                nk = len(names) - shape_dim(dist)
                worst_i = int(np.argmax(np.max(np.abs(G), axis=1)))
                wj = int(np.argmax(np.abs(G[worst_i])))
                SG = np.array([x["shape_gap_abs"] for x in v])
                print(
                    f"  {tag}: within 2 SE all {sum(x['within2_all'] for x in v)}, without omega "
                    f"{sum(x['within2_noomega'] for x in v)}; worst seed {rs[worst_i]['seed']} "
                    f"{names[wj]} {G[worst_i, wj]:+.2f} SE; median |gap| SE "
                    + ", ".join(
                        f"{nm} {np.median(np.abs(G[:, j])):.2f}" for j, nm in enumerate(names)
                    )
                )
                print(
                    "    shape gap to arch (abs): median "
                    + ", ".join(
                        f"{names[nk + j]} {np.median(np.abs(SG[:, j])):.3f}"
                        for j in range(SG.shape[1])
                    )
                    + "; max "
                    + ", ".join(f"{np.max(np.abs(SG[:, j])):.3f}" for j in range(SG.shape[1]))
                    + "; arch shape SE median "
                    + ", ".join(
                        f"{np.median([d['se'][nk + j] for d in rs]):.3f}"
                        for j in range(SG.shape[1])
                    )
                )
                print(
                    f"    sigma2 last 1000: max rel gap median over seeds "
                    f"{np.median([x['path_max_rel_last1000'] for x in v]):.4f}, max "
                    f"{np.max([x['path_max_rel_last1000'] for x in v]):.4f}; median rel gap max over seeds "
                    f"{np.max([x['path_median_rel_last1000'] for x in v]):.4f}; |persistence gap| max "
                    f"{np.max(np.abs([x['persistence_gap'] for x in v])):.4f}; ll gap median "
                    f"{np.median([x['ll_gap'] for x in v]):.2f}; starts converged "
                    f"{sum(x['start_stop'] == 'converged' for x in v)} (max it "
                    f"{max(x['start_iterations'] for x in v)}, median {np.median([x['start_seconds'] for x in v]):.2f} s); "
                    f"projected rows {sum(x['projected_rows'] for x in v)} on "
                    f"{sum(x['projected_rows'] > 0 for x in v)} seeds; us/row median "
                    f"{np.median([x['us_per_row'] for x in v]):.1f}; state {v[0]['state_floats']} floats = "
                    f"{8 * v[0]['state_floats']} bytes"
                )


# ==================================================================== FIGARCH
#
# Part 3. FIGARCH(1,1) in arch's form (the fixed-truncation Chung/Baillie-Bollerslev-Mikkelsen
# ARCH(infinity) form; arch.univariate.FIGARCH, truncation M, default 1000):
#
#     h_t = omega / (1 - beta) + sum_{j=1}^{M} lam_j e_{t-j}^2,   e_s^2 = backcast for s < 0
#     lam_1 = phi - beta + d,  delta_1 = d,
#     delta_j = (j - 1 - d)/j delta_{j-1},  lam_j = beta lam_{j-1} + delta_j - phi delta_{j-1}
#
# (arch's weights, ``recursions_python.figarch_weights_python``; the pre-sample squared residual is
# arch's backcast, so a lag j > t contributes lam_j * backcast). The truncated weights sum to less
# than 1 and the tail past M is ~ M^-d, which is why M matters (section "Truncation" of the record).
# Parameters (omega, phi, d, beta) with arch's constraints: omega > 0, phi >= 0, 0 <= d <= 1,
# 2 phi + d <= 1, 0 <= beta <= phi + d (they keep every lam_j >= 0).

FIG_NAMES = ["omega", "phi", "d", "beta"]
FIG_TRUTH = np.array([0.03, 0.2, 0.4, 0.5])
FIG_SEED0 = 492200
FIG_MS = (100, 500, 1000)
FIG_K = 4
D_MAX = 1.0 - 1e-4


def fig_weights(th: np.ndarray, m: int, deriv: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """``lam`` (m,) and, if ``deriv``, ``dlam`` (3, m): derivatives in (phi, d, beta).

    ``lam_j = beta lam_{j-1} + u_j`` with ``u_1 = d - beta + phi`` and
    ``u_j = delta_j - phi delta_{j-1}`` (j >= 2); ``delta_j = d P_j``,
    ``P_j = prod_{i=2}^{j} (i - 1 - d)/i`` (so no division by ``d``). The three derivatives are the same
    linear recursion driven by ``du/dphi``, ``du/dd`` and ``(-1, lam_1, ..., lam_{m-1})``.
    """
    _, phi, d, beta = th
    a = np.array([1.0, -beta])
    j = np.arange(2, m + 1, dtype=float)
    p = np.ones(m)
    p[1:] = np.cumprod((j - 1.0 - d) / j)
    delta = d * p
    u = np.empty(m)
    u[0] = d - beta + phi
    u[1:] = delta[1:] - phi * delta[:-1]
    lam = lfilter([1.0], a, u)
    if not deriv:
        return lam, None
    dp = np.zeros(m)
    dp[1:] = -p[1:] * np.cumsum(1.0 / (j - 1.0 - d))
    ddelta = p + d * dp
    du_phi = np.empty(m)
    du_phi[0] = 1.0
    du_phi[1:] = -delta[:-1]
    du_d = np.empty(m)
    du_d[0] = 1.0
    du_d[1:] = ddelta[1:] - phi * ddelta[:-1]
    du_b = np.empty(m)
    du_b[0] = -1.0
    du_b[1:] = lam[:-1]
    dlam = np.vstack([lfilter([1.0], a, du_phi), lfilter([1.0], a, du_d), lfilter([1.0], a, du_b)])
    return lam, dlam


def fig_constraints(floor: float) -> tuple[np.ndarray, np.ndarray]:
    """``A x >= b`` for x = (omega, phi, d, beta): arch's ``FIGARCH.constraints`` plus ``d <= 1 - 1e-4``."""
    a = np.array(
        [
            [1, 0, 0, 0],  # omega >= floor
            [0, 1, 0, 0],  # phi >= 0
            [0, 0, 1, 0],  # d >= 0
            [0, 0, -1, 0],  # d <= 1 - eps
            [0, -2, -1, 0],  # 2 phi + d <= 1
            [0, 0, 0, 1],  # beta >= 0
            [0, 1, 1, -1],  # beta <= phi + d
        ],
        dtype=float,
    )
    b = np.array([floor, 0.0, 0.0, -D_MAX, -1.0, 0.0, 0.0])
    return a, b


def fig_project(th: np.ndarray, scale: float) -> np.ndarray:
    a, b = fig_constraints(OMEGA_FLOOR_FRAC * scale)
    return project_halfspaces(np.asarray(th, dtype=float), a, b)


def fig_project_metric(th: np.ndarray, hess: np.ndarray, scale: float) -> np.ndarray:
    """The projection in the metric of ``hess`` (the step's own scaling), then the Euclidean one if rounding
    left it a hair outside."""
    a, b = fig_constraints(OMEGA_FLOOR_FRAC * scale)
    y = project_halfspaces_metric(th, a, b, hess)
    return fig_project(y, scale) if np.any(a @ y < b - 1e-12) else y


def fig_h_path(th: np.ndarray, x: np.ndarray, m: int, start: float) -> np.ndarray:
    """Conditional variances over ``x = e^2`` at fixed ``th`` (arch's filter, pre-sample ``start``)."""
    lam, _ = fig_weights(th, m, deriv=False)
    n = x.shape[0]
    ext = np.concatenate([np.full(m, start), x])
    c = np.concatenate([[0.0], lam])
    return th[0] / (1.0 - th[3]) + np.convolve(ext, c)[m : m + n]


def fig_objective(x: np.ndarray, m: int, start: float):  # noqa: ANN201
    """Gaussian log-likelihood of the rows ``sqrt(x)``, its score and Gauss-Newton matrix, in (omega, phi, d, beta)."""
    n = x.shape[0]
    ext = np.concatenate([np.full(m, start), x])

    def f(th: np.ndarray, grad: bool):  # noqa: ANN202
        bad = (-math.inf, np.zeros(FIG_K), np.zeros((FIG_K, FIG_K)))
        if not (th[0] > 0.0 and 0.0 <= th[3] < 1.0 and 0.0 <= th[2] < 1.0):
            return bad
        lam, dlam = fig_weights(th, m, deriv=grad)
        c = np.concatenate([[0.0], lam])
        h = th[0] / (1.0 - th[3]) + np.convolve(ext, c)[m : m + n]
        if not np.all(h > 0.0):
            return bad
        ll = -0.5 * float(np.sum(LN_2PI + np.log(h) + x / h))
        if not grad:
            return ll, bad[1], bad[2]
        dh = np.empty((n, FIG_K))
        dh[:, 0] = 1.0 / (1.0 - th[3])
        for i in range(3):
            dh[:, 1 + i] = np.convolve(ext, np.concatenate([[0.0], dlam[i]]))[m : m + n]
        dh[:, 3] += th[0] / (1.0 - th[3]) ** 2
        dl = dh / h[:, None]
        return ll, 0.5 * dl.T @ (x / h - 1.0), 0.5 * dl.T @ dl

    return f


def fig_grid(v: float, m: int) -> list[np.ndarray]:
    """arch's starting grid (d in .2/.5/.7, phi and beta as fractions of their bounds), omega from the variance target."""
    out = []
    for d in (0.2, 0.5, 0.7):
        for pr in (0.2, 0.5, 0.8):
            phi = (1.0 - d) / 2.0 * pr
            for br in (0.1, 0.5, 0.9):
                beta = (d + phi) * br
                lam, _ = fig_weights(np.array([0.0, phi, d, beta]), m, deriv=False)
                out.append(np.array([(1.0 - beta) * v * (1.0 - float(lam.sum())), phi, d, beta]))
    return out


@dataclass
class FigOnline:
    """Online FIGARCH(1,1): the truncated ARCH(infinity) form with a ring of the last ``m`` squared residuals.

    State: the ring (m floats), theta (4), H (4 x 4 symmetric, 10), and the running mean of e^2, its count and
    the step count (3): ``m + 17`` floats, ``8 m + 136`` bytes. The weights and their derivatives are not state:
    they are a pure function of ``theta`` and cost O(m) to produce (a Rust port would produce them in one pass
    over the ring with O(1) scratch; this script materialises four m-vectors per row).

    Per row: h = omega/(1-beta) + lam . ring; sensitivity dh = (1/(1-beta), dlam_phi . ring, dlam_d . ring,
    omega/(1-beta)^2 + dlam_beta . ring); then the part-1 step, ``g = (z^2 - 1)/(2 h) dh``,
    ``H += gamma (dh dh'/(2 h^2) - H)``, ``theta = Proj(theta + gamma H^-1 g)``, ``gamma = 1/(t + 500)``.
    """

    m: int
    metric: bool = (
        False  # project in the H metric (see fig_project_metric) instead of the Euclidean one
    )
    warm: int = WARM
    t0: float = float(WARM)
    buffer: list = field(default_factory=list)
    theta: np.ndarray | None = None
    H: np.ndarray | None = None
    ring: np.ndarray | None = None  # ring[0] the most recent e^2
    n_seen: int = 0
    n_steps: int = 0
    mean_r2: float = 0.0
    preds: list = field(default_factory=list)
    fit: object = None
    projected: int = 0
    start_seconds: float = 0.0

    def state_floats(self) -> int:
        return self.m + FIG_K + FIG_K * (FIG_K + 1) // 2 + 3

    def update(self, r: float) -> None:
        if self.theta is None:
            self.buffer.append(r)
            self.preds.append(math.nan)
            if len(self.buffer) == self.warm:
                self._start()
            return
        self._row(r, step=True)

    def _start(self) -> None:
        t = time.perf_counter()
        rows = np.asarray(self.buffer)
        x = rows * rows
        b = backcast(x)
        v = float(np.mean(x))
        grid = [tuple(fig_project(g, v)) for g in fig_grid(v, self.m)]
        self.fit = batch_core.maximize(
            fig_objective(x, self.m, b), lambda th: fig_project(th, v), grid
        )
        self.theta = np.asarray(self.fit.theta, dtype=float).copy()
        self.H = np.zeros((FIG_K, FIG_K))
        self.ring = np.full(self.m, b)
        for r in rows:
            self._row(float(r), step=False)
        self.buffer = []
        self.start_seconds = time.perf_counter() - t

    def _row(self, r: float, step: bool) -> None:
        th = self.theta
        lam, dlam = fig_weights(th, self.m)
        w = th[0] / (1.0 - th[3])
        h = w + float(lam @ self.ring)
        dh = np.empty(FIG_K)
        dh[0] = 1.0 / (1.0 - th[3])
        dh[1:] = dlam @ self.ring
        dh[3] += w / (1.0 - th[3])
        if step:
            self.preds.append(h)
        x = r * r
        self.n_seen += 1
        self.mean_r2 += (x - self.mean_r2) / self.n_seen
        dl = dh / h
        if step:
            self.n_steps += 1
            gamma = 1.0 / (self.n_steps + self.t0)
            self.H += gamma * (0.5 * np.outer(dl, dl) - self.H)
            prop = th + gamma * np.linalg.solve(
                self.H + 1e-12 * np.eye(FIG_K), 0.5 * (x / h - 1.0) * dl
            )
            if self.metric:
                new = fig_project_metric(prop, self.H + 1e-12 * np.eye(FIG_K), self.mean_r2)
            else:
                new = fig_project(prop, self.mean_r2)
            if not np.array_equal(new, prop):
                self.projected += 1
            self.theta = new
        else:
            self.H += (0.5 * np.outer(dl, dl) - self.H) / self.n_seen
        self.ring[1:] = self.ring[:-1].copy()
        self.ring[0] = x


def simulate_fig(
    th: np.ndarray, n: int, seed: int, m_true: int = 1000, burn: int = 3000
) -> np.ndarray:
    """Gaussian FIGARCH(1,1) with truncation ``m_true``, started at the truncated model's mean variance."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + burn)
    lam, _ = fig_weights(th, m_true, deriv=False)
    w = th[0] / (1.0 - th[3])
    h0 = w / (1.0 - float(lam.sum()))
    lam_rev = lam[::-1].copy()
    xs = np.full(m_true + n + burn, h0)
    out = np.empty(n + burn)
    for t in range(n + burn):
        h = w + float(lam_rev @ xs[t : t + m_true])
        out[t] = math.sqrt(h) * z[t]
        xs[m_true + t] = out[t] * out[t]
    return out[burn:]


def fig_arch_fit(r: np.ndarray, m: int):  # noqa: ANN201
    from arch.univariate import FIGARCH, Normal, ZeroMean

    am = ZeroMean(
        r, volatility=FIGARCH(p=1, q=1, truncation=m), distribution=Normal(), rescale=False
    )
    return am.fit(disp="off", options={"maxiter": 1000})


def fig_ll(th: np.ndarray, r: np.ndarray, m: int) -> float:
    x = r * r
    return fig_objective(x, m, backcast(x))(th, False)[0]


def run_fig_online(r: np.ndarray, m: int, metric: bool) -> FigOnline:
    est = FigOnline(m, metric=metric)
    t = time.perf_counter()
    for x in r:
        est.update(float(x))
    est.run_seconds = time.perf_counter() - t - est.start_seconds  # type: ignore[attr-defined]
    return est


FIG_VARIANTS = (
    "euclid",
    "metric",
)  # the projection of the step: Euclidean (as in part 1), or in the H metric


def compare_fig(
    seed: int, n: int = 10_000, ms: tuple[int, ...] = FIG_MS, m_ref: int = 1000, m_true: int = 1000
) -> dict:
    """One seed: arch at truncation ``m_ref`` (the reference) and at each ``m`` of ``ms``, and the online form
    (both projections) at each ``m``; the keys are ``euclid_M`` and ``metric_M``. The data come from a
    truncation-``m_true`` process (arch's default, 1000, unless asked)."""
    r = simulate_fig(FIG_TRUTH, n, seed, m_true=m_true)
    x = r * r
    start = backcast(x)
    out: dict = {
        "seed": seed,
        "n": n,
        "names": FIG_NAMES,
        "truth": FIG_TRUTH.tolist(),
        "ms": list(ms),
        "m_ref": m_ref,
        "m_true": m_true,
    }
    fits = {}
    for m in sorted(set(ms) | {m_ref}):
        t = time.perf_counter()
        fits[m] = fig_arch_fit(r, m)
        fits[m].seconds = time.perf_counter() - t
    ref = fits[m_ref]
    out["arch"] = {
        str(m): {
            "params": np.asarray(f.params, dtype=float).tolist(),
            "se": np.asarray(f.std_err, dtype=float).tolist(),
            "converged": int(f.convergence_flag) == 0,
            "ll": float(f.loglikelihood),
            "seconds": f.seconds,
        }
        for m, f in fits.items()
    }
    ref_p = np.asarray(ref.params, dtype=float)
    tail = slice(n - 1000, n)
    out["truncation_rel_last1000"] = {}
    for m in ms:
        # pure truncation error: arch's reference parameters filtered at m and at m_ref
        h_m = fig_h_path(ref_p, x, m, start)
        h_r = fig_h_path(ref_p, x, m_ref, start)
        out["truncation_rel_last1000"][str(m)] = float(
            np.max(np.abs(h_m[tail] - h_r[tail]) / h_r[tail])
        )
        for variant in FIG_VARIANTS:
            est = run_fig_online(r, m, variant == "metric")
            th = est.theta
            pred = np.asarray(est.preds)
            res = {}
            for tag, f in (("same_m", fits[m]), ("ref", ref)):
                bp = np.asarray(f.params, dtype=float)
                se = np.asarray(f.std_err, dtype=float)
                path_b = np.asarray(f.conditional_volatility, dtype=float) ** 2
                rel = np.abs(pred[tail] - path_b[tail]) / path_b[tail]
                gap = ((th - bp) / se).tolist()
                res[tag] = {
                    "gap_se": gap,
                    "gap_abs": (th - bp).tolist(),
                    "within2_all": bool(np.all(np.abs(gap) <= 2.0)),
                    "within2_noomega": bool(np.all(np.abs(gap[1:]) <= 2.0)),
                    "path_max_rel_last1000": float(np.max(rel)),
                    "path_median_rel_last1000": float(np.median(rel)),
                    "ll_gap": fig_ll(th, r, m) - float(f.loglikelihood),
                }
            res.update(
                {
                    "x": th.tolist(),
                    "start_stop": est.fit.stop,
                    "start_iterations": est.fit.iterations,
                    "start_seconds": est.start_seconds,
                    "us_per_row": 1e6 * est.run_seconds / (n - WARM),  # type: ignore[attr-defined]
                    "projected_rows": est.projected,
                    "state_floats": est.state_floats(),
                    "silent_rows": int(np.sum(np.isnan(pred))),
                }
            )
            out[f"{variant}_{m}"] = res
    return out


def truncation_study(
    seed: int = 492299,
    n: int = 10_000,
    m_true: int = 32_000,
    ms: tuple[int, ...] = (100, 500, 1000, 4000, 16_000, 32_000),
) -> dict:
    """The truncation error alone, at the true parameters: data from a truncation-``m_true`` process, filtered at
    each ``m`` with the same parameters; the relative gap to the ``m_true`` filter over the last 1000 rows, and
    the weight mass beyond ``m`` (``1 - sum_{j<=m} lam_j``, since the untruncated weights sum to 1)."""
    r = simulate_fig(FIG_TRUTH, n, seed, m_true=m_true)
    x = r * r
    start = backcast(x)
    ref = fig_h_path(FIG_TRUTH, x, m_true, start)
    tail = slice(n - 1000, n)
    out = {}
    for m in ms:
        h = fig_h_path(FIG_TRUTH, x, m, start)
        lam, _ = fig_weights(FIG_TRUTH, m, deriv=False)
        rel = np.abs(h[tail] - ref[tail]) / ref[tail]
        out[m] = {
            "max_rel": float(np.max(rel)),
            "median_rel": float(np.median(rel)),
            "tail_mass": 1.0 - float(lam.sum()),
        }
    return out


def check_fig() -> None:
    """Weights against arch's, analytic derivatives against central differences, the filter against arch's, the
    objective's score against a central difference."""
    from arch.univariate.recursions_python import figarch_weights_python

    th = np.array([0.03, 0.2, 0.4, 0.5])
    lam, dlam = fig_weights(th, 1000)
    ref = figarch_weights_python(th[1:], 1, 1, 1000)
    print(f"figarch: weights vs arch max abs {np.max(np.abs(lam - ref)):.1e}")
    assert np.max(np.abs(lam - ref)) < 1e-13
    for i, nm in enumerate(("phi", "d", "beta")):
        e = np.zeros(FIG_K)
        h = 1e-6
        e[1 + i] = h
        num = (fig_weights(th + e, 1000, False)[0] - fig_weights(th - e, 1000, False)[0]) / (2 * h)
        rel = np.max(np.abs(dlam[i] - num) / np.maximum(np.abs(num), 1e-12))
        print(f"figarch: d lam / d {nm} vs central difference, max rel {rel:.1e}")
        assert rel < 1e-6
    r = simulate_fig(th, 3000, 8)
    x = r * r
    res = fig_arch_fit(r, 1000)
    ours = fig_h_path(np.asarray(res.params, dtype=float), x, 1000, backcast(x))
    theirs = np.asarray(res.conditional_volatility) ** 2
    relp = float(np.max(np.abs(ours - theirs) / theirs))
    print(f"figarch: filter vs arch at arch's parameters (every row) max rel {relp:.1e}")
    assert relp < 1e-10
    r = simulate_fig(th, 800, 7)
    x = r * r
    f = fig_objective(x, 300, backcast(x))
    _, g, _ = f(th, True)
    num = np.empty(FIG_K)
    for j in range(FIG_K):
        h = 1e-6 * max(abs(th[j]), 1e-2)
        e = np.zeros(FIG_K)
        e[j] = h
        num[j] = (f(th + e, False)[0] - f(th - e, False)[0]) / (2 * h)
    relg = float(np.max(np.abs(g - num) / np.maximum(np.abs(num), 1e-3)))
    print(f"figarch: score vs central difference of the likelihood max rel {relg:.1e}")
    assert relg < 1e-5


# ======================================================================== runner

FAMILIES = ("garch", "t", "figarch")

# Tolerances (docs/records/STUDY-492-GARCH-FAMILY.md, section "Tolerances"): each one is a count or a bound
# taken from the measured distribution over seeds, then loosened (1.5 times the worst, to a clean figure).
TOL: dict[str, float] = {}


def case_keys(family: str, seeds: int, models: list[str] | None = None) -> list[tuple]:
    if family == "garch":
        return [(m, SEED0 + i) for i in range(seeds) for m in (models or list(MODELS))]
    if family == "t":
        return [
            (m, d, T_SEED0 + i)
            for i in range(seeds)
            for m, (_, _, dists) in T_MODELS.items()
            for d in dists
            if (not models or f"{m}:{d}" in models)
            and (models or m != "g22")  # g22 t is wired, not run
        ]
    return [(FIG_SEED0 + i,) for i in range(seeds)]


def run_case(
    family: str, key: tuple, n: int, ms: tuple[int, ...] = FIG_MS, m_true: int = 1000
) -> dict:
    if family == "garch":
        return compare_garch(key[0], key[1], n)
    if family == "t":
        return compare_t(key[0], key[1], key[2], n)
    return compare_fig(key[0], n, ms=ms, m_ref=ms[-1], m_true=m_true)


def key_of(family: str, d: dict) -> tuple:
    if family == "garch":
        return (d["model"], d["seed"])
    if family == "t":
        return (d["model"], d["dist"], d["seed"])
    return (d["seed"],)


def load_rows(family: str, outdir: str) -> list[dict]:
    rows: list[dict] = []
    for p in Path(outdir).glob("cases_*.jsonl"):
        rows += [json.loads(x) for x in p.read_text().splitlines()]
    return rows


def run_grid(
    family: str,
    outdir: str,
    seeds: int,
    n: int,
    models: list[str] | None = None,
    ms: tuple[int, ...] = FIG_MS,
    m_true: int = 1000,
) -> list[dict]:
    """Run the family's cases, one JSON line each, resumable (the cases already on disk are skipped) and
    shardable (SHARD of NSHARDS). Returns the rows run by this process."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    done = {key_of(family, d) for d in load_rows(family, outdir)}
    shard = int(os.environ.get("SHARD", "0"))
    nshards = int(os.environ.get("NSHARDS", "1"))
    mine = []
    with (out / f"cases_{os.getpid()}.jsonl").open("a") as fh:
        for i, key in enumerate(case_keys(family, seeds, models)):
            if i % nshards != shard or key in done:
                continue
            d = run_case(family, key, n, ms, m_true)
            fh.write(json.dumps(d) + "\n")
            fh.flush()
            mine.append(d)
            print(f"done {family} {key}", flush=True)
    return mine


# ---------------------------------------------------------------------- figarch summary


def summary_fig(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda d: d["seed"])
    if not rows:
        return
    ms = rows[0]["ms"]
    m_ref = rows[0]["m_ref"]
    print(
        f"\nFIGARCH(1,1): {len(rows)} seeds, n {rows[0]['n']}, truth {rows[0]['truth']}, data from truncation "
        f"{rows[0].get('m_true', 1000)}, reference arch truncation {m_ref}"
    )
    for m in sorted(set(ms) | {m_ref}):
        a = [d["arch"][str(m)] for d in rows]
        P = np.array([x["params"] for x in a])
        print(
            f"  arch M={m}: converged {sum(x['converged'] for x in a)}, fit median {np.median([x['seconds'] for x in a]):.1f} s, "
            "mean "
            + ", ".join(f"{nm} {P[:, j].mean():.3f}" for j, nm in enumerate(FIG_NAMES))
            + "; SE median "
            + ", ".join(f"{np.median([x['se'][j] for x in a]):.3f}" for j in range(FIG_K))
        )
    for m in ms:
        tr = np.array([d["truncation_rel_last1000"][str(m)] for d in rows])
        print(
            f"  filter truncation alone (arch M={m_ref} parameters, M={m} against M={m_ref}, last 1000 rows): "
            f"max rel median {np.median(tr):.4f}, worst {tr.max():.4f}"
        )
    for variant in FIG_VARIANTS:
        for m in ms:
            v = [d[f"{variant}_{m}"] for d in rows]
            print(
                f"  online {variant} M={m}: state {v[0]['state_floats']} floats = {8 * v[0]['state_floats']} bytes "
                f"(8 M + 136), {np.median([x['us_per_row'] for x in v]):.0f} us/row median, start "
                f"{sum(x['start_stop'] == 'converged' for x in v)}/{len(v)} converged (max it "
                f"{max(x['start_iterations'] for x in v)}, median {np.median([x['start_seconds'] for x in v]):.2f} s), "
                f"projected rows {sum(x['projected_rows'] for x in v)} on {sum(x['projected_rows'] > 0 for x in v)} seeds"
            )
            for tag, lab in (("same_m", f"arch at M={m}"), ("ref", f"arch at M={m_ref}")):
                if tag == "ref" and m == m_ref:
                    continue
                G = np.array([x[tag]["gap_se"] for x in v])
                A = np.array([x[tag]["gap_abs"] for x in v])
                worst_i = int(np.argmax(np.max(np.abs(G), axis=1)))
                wj = int(np.argmax(np.abs(G[worst_i])))
                pm = np.array([x[tag]["path_max_rel_last1000"] for x in v])
                pd = np.array([x[tag]["path_median_rel_last1000"] for x in v])
                ll = np.array([x[tag]["ll_gap"] for x in v])
                print(
                    f"    vs {lab}: within 2 SE all {sum(x[tag]['within2_all'] for x in v)}, without omega "
                    f"{sum(x[tag]['within2_noomega'] for x in v)} of {len(v)}; worst seed {rows[worst_i]['seed']} "
                    f"{FIG_NAMES[wj]} {G[worst_i, wj]:+.2f} SE"
                )
                print(
                    "      median |gap| (abs) "
                    + ", ".join(
                        f"{nm} {np.median(np.abs(A[:, j])):.4f}" for j, nm in enumerate(FIG_NAMES)
                    )
                    + "; median |gap| (SE) "
                    + ", ".join(
                        f"{nm} {np.median(np.abs(G[:, j])):.2f}" for j, nm in enumerate(FIG_NAMES)
                    )
                )
                print(
                    f"      sigma2 last 1000 rows: max-rel-gap median over seeds {np.median(pm):.4f}, worst {pm.max():.4f}; "
                    f"median-rel-gap median {np.median(pd):.4f}, worst {pd.max():.4f}; ll gap median "
                    f"{np.median(ll):.2f}, worst {ll.min():.2f}"
                )


# ---------------------------------------------------------------------- verdicts


def verdicts(family: str, rows: list[dict], quick: bool) -> list[tuple[str, bool, str]]:
    """The tolerances of the record, each as (name, passed, measured against bound). ``quick`` swaps in the
    smoke bounds (a few seeds, short series): structural only, so it cannot flake on a 2 SE count."""
    out: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, text: str) -> None:
        out.append((name, bool(ok), text))

    if family == "garch":
        for model in ("garch11", "gjr", "egarch"):
            rs = [d for d in rows if d["model"] == model]
            if not rs:
                continue
            v = [d["plain"] for d in rs]
            n = len(v)
            pm = np.array([x["path_max_rel_last1000"] for x in v])
            c2 = sum(x["within2_all"] for x in v)
            if quick:
                add(
                    f"{model}: variance path gap",
                    float(np.median(pm)) < TOL["quick_path"],
                    f"median over seeds of the max {np.median(pm):.3f} < {TOL['quick_path']}",
                )
                add(
                    f"{model}: starts ran",
                    all(x["start_iterations"] > 0 for x in v),
                    "batch start iterated",
                )
            else:
                add(
                    f"{model}: within 2 SE of arch (every parameter)",
                    c2 >= math.ceil(TOL["garch_within2_frac"] * n),
                    f"{c2}/{n} seeds, bound >= {TOL['garch_within2_frac']:.2f} of seeds",
                )
                add(
                    f"{model}: variance path gap, median over seeds of the max over the last 1000 rows",
                    float(np.median(pm)) <= TOL["garch_path_median"],
                    f"{np.median(pm):.3f}, bound {TOL['garch_path_median']}",
                )
                if (
                    f"{model}_path_worst" in TOL
                ):  # EGARCH has no bound on the worst seed (61%: omega)
                    add(
                        f"{model}: variance path gap, worst seed",
                        float(pm.max()) <= TOL[f"{model}_path_worst"],
                        f"{pm.max():.3f}, bound {TOL[f'{model}_path_worst']}",
                    )
        g22 = [d for d in rows if d["model"] == "garch22"]
        if g22 and not quick:
            c2 = sum(d["plain"]["within2_all"] for d in g22)
            add(
                "garch22: no pass tolerance (the failure is the expected result: beta1 and beta2 are weakly "
                "identified)",
                True,
                f"{c2}/{len(g22)} within 2 SE, recorded only",
            )
    elif family == "t":
        for (model, dist), lab in (
            (("g11", "t"), "GARCH(1,1) t"),
            (("g11", "skewt"), "GARCH(1,1) skew-t"),
            (("gjr", "t"), "GJR t"),
        ):
            rs = [d for d in rows if d["model"] == model and d["dist"] == dist]
            if not rs:
                continue
            v = [d["plain_online"] for d in rs]
            n = len(v)
            m = shape_dim(dist)
            nk = len(rs[0]["names"]) - m
            G = np.abs(np.array([x["gap_se"] for x in v]))[:, nk:]
            pm = np.array([x["path_max_rel_last1000"] for x in v])
            c2 = sum(x["within2_all"] for x in v)
            s2 = int(np.sum(np.all(G <= 2.0, axis=1)))
            med = np.median(G, axis=0)
            if quick:
                add(
                    f"{lab}: median shape gap",
                    float(med.max()) < TOL["quick_shape_se"],
                    f"{med.max():.2f} arch SE < {TOL['quick_shape_se']}",
                )
                add(
                    f"{lab}: median path gap",
                    float(np.median(pm)) < TOL["quick_path"],
                    f"{np.median(pm):.3f} < {TOL['quick_path']}",
                )
            else:
                add(
                    f"{lab}: within 2 SE of arch (every parameter)",
                    c2 >= math.ceil(TOL["t_within2_frac"] * n),
                    f"{c2}/{n} seeds, bound >= {TOL['t_within2_frac']:.2f}",
                )
                add(
                    f"{lab}: shape within 2 SE of arch",
                    s2 >= math.ceil(TOL["t_shape_within2_frac"] * n),
                    f"{s2}/{n} seeds, bound >= {TOL['t_shape_within2_frac']:.2f}",
                )
                add(
                    f"{lab}: shape gap, median over seeds, each shape parameter",
                    float(med.max()) <= TOL["t_shape_median_se"],
                    f"{', '.join(f'{x:.2f}' for x in med)} arch SE, bound {TOL['t_shape_median_se']}",
                )
                add(
                    f"{lab}: variance path gap, median over seeds of the max (last 1000 rows)",
                    float(np.median(pm)) <= TOL["t_path_median"],
                    f"{np.median(pm):.3f}, bound {TOL['t_path_median']}",
                )
                p90 = float(np.percentile(pm, 90))
                add(
                    f"{lab}: variance path gap, 90th percentile over seeds (no bound on the worst: blown-up seeds)",
                    p90 <= TOL["t_path_p90"],
                    f"{p90:.3f}, bound {TOL['t_path_p90']}; worst {pm.max():.2f}",
                )
    else:
        for m in rows[0]["ms"] if rows else []:
            v = [d[f"metric_{m}"] for d in rows]
            pm = np.array([x["same_m"]["path_max_rel_last1000"] for x in v])
            if quick:
                add(
                    f"figarch M={m}: variance path gap",
                    float(np.median(pm)) < TOL["quick_path"],
                    f"median over seeds of the max {np.median(pm):.3f} < {TOL['quick_path']}",
                )
                add(
                    f"figarch M={m}: state",
                    v[0]["state_floats"] == m + 17,
                    f"{v[0]['state_floats']} floats = M + 17",
                )
            elif m == TOL["fig_m"]:
                c2 = sum(x["same_m"]["within2_all"] for x in v)
                G = np.abs(np.array([x["same_m"]["gap_se"] for x in v]))
                add(
                    f"figarch M={m}: within 2 SE of arch at the same M",
                    c2 >= math.ceil(TOL["fig_within2_frac"] * len(v)),
                    f"{c2}/{len(v)} seeds, bound >= {TOL['fig_within2_frac']:.2f}",
                )
                add(
                    f"figarch M={m}: parameter gap, median over seeds, each parameter",
                    float(np.median(G, axis=0).max()) <= TOL["fig_gap_median_se"],
                    f"{', '.join(f'{x:.2f}' for x in np.median(G, axis=0))} arch SE, bound {TOL['fig_gap_median_se']}",
                )
                add(
                    f"figarch M={m}: variance path gap, median over seeds of the max (last 1000 rows)",
                    float(np.median(pm)) <= TOL["fig_path_median"],
                    f"{np.median(pm):.3f}, bound {TOL['fig_path_median']}",
                )
                add(
                    f"figarch M={m}: variance path gap, worst seed",
                    float(pm.max()) <= TOL["fig_path_worst"],
                    f"{pm.max():.3f}, bound {TOL['fig_path_worst']}",
                )
                add(
                    f"figarch M={m}: no seed leaves the region (log-likelihood within 50 of arch's)",
                    min(x["same_m"]["ll_gap"] for x in v) > -50.0,
                    f"worst log-likelihood gap {min(x['same_m']['ll_gap'] for x in v):.1f}",
                )
    return out


def report(family: str, rows: list[dict], quick: bool) -> bool:
    if family == "garch":
        summary_garch_rows(rows)
    elif family == "t":
        summary_t_rows(rows)
    else:
        summary_fig(rows)
    ok = True
    print(f"\n{family}: tolerance verdicts{' (quick: smoke bounds)' if quick else ''}")
    for name, passed, text in verdicts(family, rows, quick):
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {text}")
        ok &= passed
    return ok


def run_family(
    family: str,
    quick: bool,
    seeds: int | None,
    n: int | None,
    out: str | None,
    models: list[str] | None,
    ms: tuple[int, ...] | None,
    m_true: int = 1000,
) -> bool:
    import tempfile

    q = QUICK[family]
    seeds = seeds if seeds is not None else (q["seeds"] if quick else DEFAULT_SEEDS[family])
    n = n if n is not None else (q["n"] if quick else 10_000)
    models = models if models is not None else (q.get("models") if quick else None)
    ms = ms if ms is not None else (q["ms"] if quick and "ms" in q else FIG_MS)
    t = time.perf_counter()
    {"garch": check_garch, "t": check_t, "figarch": check_fig}[family]()
    with tempfile.TemporaryDirectory() as tmp:
        outdir = out or (tmp if quick else f".cache/study492_{family}")
        run_grid(family, outdir, seeds, n, models, ms, m_true)
        rows = load_rows(family, outdir)
        ok = report(family, rows, quick)
    print(f"{family}: {time.perf_counter() - t:.0f} s")
    return ok


DEFAULT_SEEDS = {"garch": 50, "t": 50, "figarch": 10}
QUICK = {
    "garch": {"seeds": 3, "n": 3000, "models": ["garch11", "gjr"]},
    "t": {"seeds": 2, "n": 3000, "models": ["g11:t"]},
    "figarch": {"seeds": 2, "n": 3000, "ms": (100, 300)},
}
TOL.update(
    {
        "quick_path": 1.0,
        "quick_shape_se": 3.0,
        # Gaussian GARCH(1,1) and GJR, 50 seeds: counts as fractions of seeds, gaps relative
        "garch_within2_frac": 0.90,  # measured 49/50 for GARCH(1,1) and GJR, 46/50 for EGARCH (see below)
        "garch_path_median": 0.05,  # measured medians 0.025 and 0.034, times 1.5
        "garch11_path_worst": 0.6,  # measured worst 0.363 (one seed, omega 3.7 SE high), times 1.5, rounded up
        "gjr_path_worst": 0.2,  # measured worst 0.111, times 1.5, rounded up
        # EGARCH has the count and the median bound only (its worst seed, 61%, is omega on the persistence ridge)
        # Student-t and skew-t, 50 seeds
        "t_within2_frac": 0.80,  # measured 46, 43, 48 of 50 (0.86 at the least)
        "t_shape_within2_frac": 0.85,  # measured 47, 46, 49 of 50
        "t_shape_median_se": 0.5,  # measured medians 0.25, 0.34 (eta), 0.12 (lambda), 0.29; arch SE units
        "t_path_median": 0.10,  # measured medians 0.035, 0.041, 0.055, times 1.5
        "t_path_p90": 0.25,  # measured 90th percentiles 0.115, 0.168, 0.088, times 1.5
        # FIGARCH, metric projection, truncation 1000 (arch's, the one chosen), 20 seeds
        "fig_m": 1000,
        "fig_within2_frac": 0.75,  # measured 17/20
        "fig_gap_median_se": 1.0,  # measured medians at most 0.5 arch SE
        "fig_path_median": 0.10,  # measured median 0.050, times 1.5, rounded up
        "fig_path_worst": 0.25,  # measured worst 0.169, times 1.5, rounded up
    }
)


def main(argv: list[str]) -> int:
    cmds = {"garch", "t", "figarch", "all", "check", "one", "grid", "summary"}
    if not argv or argv[0] not in cmds:
        argv = ["all", *argv]
    ap = argparse.ArgumentParser(prog="study_492.py", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=sorted(cmds))
    ap.add_argument("args", nargs="*")
    ap.add_argument(
        "--quick", action="store_true", help="a few seeds on short series; under two minutes"
    )
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--n", type=int)
    ap.add_argument(
        "--out",
        help="grid directory (default .cache/study492_FAMILY; a temporary one with --quick)",
    )
    ap.add_argument("--models", help="comma-separated, e.g. gjr,garch11 or g11:t,g11:skewt")
    ap.add_argument(
        "--m-true", type=int, default=1000, help="FIGARCH: the truncation of the simulated process"
    )
    ap.add_argument(
        "--ms", help="FIGARCH truncation lengths, comma-separated (the last is the reference)"
    )
    a = ap.parse_args(argv)
    models = a.models.split(",") if a.models else None
    ms = tuple(int(x) for x in a.ms.split(",")) if a.ms else None
    if a.cmd in ("garch", "t", "figarch", "all"):
        fams = FAMILIES if a.cmd == "all" else (a.cmd,)
        t0 = time.perf_counter()
        oks = [run_family(f, a.quick, a.seeds, a.n, a.out, models, ms, a.m_true) for f in fams]
        print(
            f"\nstudy_492 {a.cmd}{' --quick' if a.quick else ''}: {'PASS' if all(oks) else 'FAIL'} "
            f"in {time.perf_counter() - t0:.0f} s"
        )
        return 0 if all(oks) else 1
    if a.cmd == "check":
        fam = a.args[0] if a.args else "all"
        for f in FAMILIES if fam == "all" else (fam,):
            {"garch": check_garch, "t": check_t, "figarch": check_fig}[f]()
        return 0
    if a.cmd == "one":
        fam, rest = a.args[0], a.args[1:]
        n = a.n or 10_000
        if fam == "garch":
            print(json.dumps(compare_garch(rest[0], int(rest[1]), n), indent=1))
        elif fam == "t":
            print(json.dumps(compare_t(rest[0], rest[1], int(rest[2]), n), indent=1))
        else:
            print(
                json.dumps(
                    compare_fig(
                        int(rest[0]), n, ms=ms or FIG_MS, m_ref=(ms or FIG_MS)[-1], m_true=a.m_true
                    ),
                    indent=1,
                )
            )
        return 0
    fam, outdir = a.args[0], a.args[1]
    if a.cmd == "grid":
        run_grid(
            fam,
            outdir,
            a.seeds or DEFAULT_SEEDS[fam],
            a.n or 10_000,
            models,
            ms or FIG_MS,
            a.m_true,
        )
    else:
        report(fam, load_rows(fam, outdir), False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
