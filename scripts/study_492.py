"""Study 492, part 1: online GJR-GARCH(1,1), EGARCH(1,1) and GARCH(2,2) with
Gaussian innovations against arch's batch fits on the same rows, numpy only
(docs/records/STUDY-492-GARCH-FAMILY.md).

Every model is filtered in one of two forms, with ``l = log sigma2``:

    GARCH(p, o, q) (arch's notation: p ARCH lags, o asymmetric lags, q GARCH lags)
        sigma2_t = omega + sum_i a_i e2_{t-i} + sum_j g_j 1[e_{t-j} < 0] e2_{t-j}
                         + sum_k b_k sigma2_{t-k}
        dsigma2_t = (1, e2 lags, 1[e<0] e2 lags, sigma2 lags) + sum_k b_k dsigma2_{t-k}
        dl_t = dsigma2_t / sigma2_t
    EGARCH(1, 1, 1), z = e / sigma, c = E|z| = sqrt(2 / pi)
        l_t  = omega + a (|z_{t-1}| - c) + g z_{t-1} + b l_{t-1}
        dl_t = (1, |z_{t-1}| - c, z_{t-1}, l_{t-1}) + (b - (a |z_{t-1}| + g z_{t-1}) / 2) dl_{t-1}

and each takes the same recursive quasi-maximum-likelihood step as the base's
GARCH(1,1) (scripts/study_485/garch.py), on the score of the Gaussian
quasi-likelihood ``l_t = -1/2 (ln 2 pi + l_t + e_t^2 / sigma2_t)``:

    g_t = 1/2 (z_t^2 - 1) dl_t,   I_t = 1/2 dl_t dl_t'
    H_t = H_{t-1} + (I_t - H_{t-1}) / (t + t0)
    theta = Proj(theta + H_t^{-1} g_t / (t + t0))

``(z^2 - 1)`` is replaced by the base's Huber-clipped residual when
``robust_c > 0`` (running centring and slope). ``Proj`` is the Euclidean
projection (Dykstra's algorithm over half-spaces) onto arch's own constraint
set, with the base's ``1 - 1e-4`` stationarity margin and ``omega`` floor.

The start is the base's adopted one (study 485, request A12): the first 500
rows are buffered and emit nothing; the core-held projected Newton optimizer
(``batch_core.maximize``) fits them from a small grid, the buffer is replayed at
that fit to build ``H`` as an equal-weight mean, and steps run from row 501 at
``1 / (t + 500)``. arch is the oracle only.

    study_492.py check                     # gradient and filter checks (seconds)
    study_492.py one MODEL SEED [N]        # one case, printed
    study_492.py grid OUTDIR [SEEDS] [N]   # one JSON line per (seed, model), resumable
    study_492.py summary OUTDIR
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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
            "e": [math.sqrt(start)] * m,  # pre-sample e2 = start, sign: positive
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
            x[1 + p + j] = e[j] * e[j] if e[j] < 0.0 else 0.0
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
        st["s2"] = [s2] + st["s2"][:-1]
        st["d"] = [dl * s2] + st["d"][:-1]

    def state_floats(self) -> int:
        k = self.k
        return k + k * (k + 1) // 2 + max(self.p, self.o) + self.q + self.q * k + 5


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
        grid = [tuple(spec.project(g, v)) for g in spec.grid(v)]
        self.fit = batch_core.maximize(objective(spec, rows, b), lambda th: spec.project(th, v), grid)
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
        info = 0.5 * np.outer(dl, dl)
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
            proposal = self.theta + gamma * np.linalg.solve(self.H, 0.5 * resid * dl)
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


def compare(model: str, seed: int, n: int = 10_000) -> dict:
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
    for tag, c in (("plain", 0.0), ("huber2", 2.0)):
        est = Online(spec, robust_c=c)
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
        ll_on = -0.5 * float(np.sum(LN_2PI + np.log(fp := filter_path(spec, th, r, start)) + r * r / fp))
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


def check() -> None:
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
        print(f"{model}: score vs central difference max rel {relg:.2e}; "
              f"filter vs arch after row 500 max rel {relp:.2e}; state {spec.state_floats()} floats")
        assert relg < 1e-5, relg


def grid(outdir: str, seeds: int = 50, n: int = 10_000) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"cases_{os.getpid()}.jsonl"
    done = set()
    for p in out.glob("cases_*.jsonl"):
        for line in p.read_text().splitlines():
            d = json.loads(line)
            done.add((d["model"], d["seed"]))
    shard = int(os.environ.get("SHARD", "0"))
    nshards = int(os.environ.get("NSHARDS", "1"))
    with path.open("a") as fh:
        for i in range(seeds):
            if i % nshards != shard:
                continue
            for model in MODELS:
                if (model, SEED0 + i) in done:
                    continue
                d = compare(model, SEED0 + i, n)
                fh.write(json.dumps(d) + "\n")
                fh.flush()


def summary(outdir: str) -> None:
    rows = []
    for p in Path(outdir).glob("cases_*.jsonl"):
        rows += [json.loads(x) for x in p.read_text().splitlines()]
    for model, (spec, _) in MODELS.items():
        rs = sorted((d for d in rows if d["model"] == model), key=lambda d: d["seed"])
        if not rs:
            continue
        print(f"\n{spec.name}: {len(rs)} seeds, n {rs[0]['n']}, arch converged "
              f"{sum(d['arch_converged'] for d in rs)}, arch fit median "
              f"{np.median([d['arch_seconds'] for d in rs]):.2f} s")
        names = rs[0]["names"]
        for tag in ("plain", "huber2"):
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
        print(f"  state: {spec.state_floats()} floats, {8 * spec.state_floats()} bytes (+ 500-row buffer, 4 KB, until the start)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "one":
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 10_000
        print(json.dumps(compare(sys.argv[2], int(sys.argv[3]), n), indent=1))
    elif cmd == "grid":
        grid(sys.argv[2], *(int(x) for x in sys.argv[3:]))
    elif cmd == "summary":
        summary(sys.argv[2])
