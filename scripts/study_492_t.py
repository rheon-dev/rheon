"""Study 492, part 2: online GARCH with Student-t and skew-t (Hansen 1994)
innovations, the degrees of freedom nu (and the skew lambda) fitted in the batch
start, against arch's batch fits on the same rows. numpy only; arch and scipy
are the oracle (docs/records/STUDY-492-GARCH-FAMILY.md, part 2).

The innovations ``z`` are standardised (mean 0, variance 1), ``e_t = sigma_t z_t``:

    Student-t    log f(z) = ln c - (nu + 1)/2 ln(1 + z^2/(nu - 2)),
                 c = G((nu+1)/2) / (sqrt(pi (nu - 2)) G(nu/2)),  nu > 2
    skew-t       log f(z) = ln b + ln c - (nu + 1)/2 ln(1 + m^2/(nu - 2)),
                 m = (b z + a)/(1 + sg lambda),  sg = sign(z + a/b),
                 a = 4 lambda c (nu - 2)/(nu - 1),  b^2 = 1 + 3 lambda^2 - a^2

Per-row log-likelihood ``l_t = log f(z_t) - 1/2 ln s_t`` (``s_t`` the one-step
variance). Score in the GARCH parameters ``theta`` (``dl_t = d ln s_t/d theta``):

    g_t = 1/2 w_t dl_t,   Student-t:  w = (nu + 1) z^2/(nu - 2 + z^2) - 1
                          skew-t:     w = (nu + 1) m b z/((1 + sg lambda)(nu - 2 + m^2)) - 1

The shape score is d log f/d nu (closed form for the t; a central difference of
the closed-form density for the skew-t, nu and lambda), the online step takes
BHHH information (the running mean of the outer product of the whole score).

    study_492_t.py check                       # scores and densities against arch (seconds)
    study_492_t.py one MODEL DIST SEED [N]     # one case, printed
    study_492_t.py grid OUTDIR [SEEDS] [N]     # one JSON line per (seed, model, dist), resumable
    study_492_t.py summary OUTDIR
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import digamma, gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "study_485"))
import batch_core  # noqa: E402
from study_492 import GarchPOQ, backcast  # noqa: E402

WARM = 500
SEED0 = 492100
NU_LO, NU_HI = 2.05, 300.0  # arch's bounds
LAM_LIM = 0.99


# ----------------------------------------------------------------- the densities


def consts(nu: float, lam: float) -> tuple[float, float, float]:
    c = math.exp(math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(math.pi * (nu - 2.0))
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
                tuple(project(np.concatenate([g[1:], s[:m]])))
                for g in spec.grid(v)
                for s in shapes
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

MODELS = {
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

    return np.asarray(
        SkewStudent(seed=np.random.default_rng(seed)).simulate(TRUE_SHAPE[dist])(n)
    )


def simulate(spec: GarchPOQ, theta: np.ndarray, dist: str, n: int, seed: int, burn: int = 1000):
    z = draws(dist, n + burn, seed)
    st = spec.init(1.0)
    out = np.empty(n + burn)
    for t in range(n + burn):
        s2, dl = spec.predict(theta, st)
        out[t] = math.sqrt(s2) * z[t]
        spec.push(st, out[t], s2, dl)
    return out[burn:]


def arch_fit(spec: GarchPOQ, dist: str, r: np.ndarray):  # noqa: ANN201
    from arch import arch_model

    am = arch_model(r, mean="Zero", dist=dist, rescale=False, **spec.arch_kwargs())
    return am.fit(disp="off", options={"maxiter": 1000})


def loglik(spec: GarchPOQ, dist: str, x: np.ndarray, r: np.ndarray) -> float:
    return make_objective(spec, dist, r, backcast(r * r))(x, False)[0]


def compare(model: str, dist: str, seed: int, n: int = 10_000) -> dict:
    spec, truth, _ = MODELS[model]
    r = simulate(spec, truth, dist, n, seed)
    t = time.perf_counter()
    res = arch_fit(spec, dist, r)
    arch_s = time.perf_counter() - t
    bp = np.asarray(res.params, dtype=float)
    se = np.asarray(res.std_err, dtype=float)
    path_b = np.asarray(res.conditional_volatility, dtype=float) ** 2
    names = list(res.params.index)
    k = spec.k
    out = {
        "model": model, "dist": dist, "seed": seed, "n": n, "names": names,
        "truth": np.concatenate([truth, TRUE_SHAPE[dist]]).tolist(),
        "batch": bp.tolist(), "se": se.tolist(),
        "arch_converged": int(res.convergence_flag) == 0,
        "arch_seconds": arch_s, "ll_batch": float(res.loglikelihood),
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
                "x": full.tolist(), "gap_se": gap,
                "within2_all": bool(np.all(np.abs(gap) <= 2.0)),
                "within2_noomega": bool(np.all(np.abs(gap[1:]) <= 2.0)),
                "shape_gap_abs": (full[k:] - bp[k:]).tolist(),
                "shape_start": est.shape.tolist() if not upd else None,
                "path_max_rel_last1000": float(np.max(rel)),
                "path_median_rel_last1000": float(np.median(rel)),
                "persistence_gap": spec.persistence(full[:k]) - spec.persistence(bp[:k]),
                "ll_gap": loglik(spec, dist, full, r) - float(res.loglikelihood),
                "start_stop": est.fit.stop, "start_iterations": est.fit.iterations,
                "start_seconds": est.start_seconds,
                "us_per_row": 1e6 * secs / (n - WARM),
                "projected_rows": est.projected,
                "state_floats": est.state_floats(),
            }
    return out


# ------------------------------------------------------------------------ check


def check() -> None:
    from scipy import integrate

    spec, truth, _ = MODELS["g11"]
    for dist in ("t", "skewt"):
        r = simulate(spec, truth, dist, 800, 7)
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
        print(f"{dist}: score (theta, shape) vs central difference of the likelihood, max rel "
              f"{rel.max():.2e} (theta {rel[:spec.k].max():.2e}, shape {rel[spec.k:].max():.2e})")
        assert rel[: spec.k].max() < 1e-5
        # likelihood at arch's own parameters (exact, with constants) against arch's
        res = arch_fit(spec, dist, r)
        ours = loglik(spec, dist, np.asarray(res.params), r)
        print(f"{dist}: log-likelihood at arch's parameters: ours - arch = {ours - res.loglikelihood:.2e}")
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
    print(f"t: shape score closed form vs difference max abs {np.max(np.abs(S[:, 0] - Sn)):.2e}, mean {S.mean():.2e}")
    zs = draws("skewt", 400_000, 3)
    _, S2 = scores("skewt", zs, 6.0, -0.2)
    print(f"skew-t: mean score (nu, lambda) at truth {S2.mean(axis=0)}, SE {S2.std(axis=0) / math.sqrt(len(zs))}")


def grid(outdir: str, seeds: int = 10, n: int = 10_000) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"cases_{os.getpid()}.jsonl"
    done = set()
    for p in out.glob("cases_*.jsonl"):
        for line in p.read_text().splitlines():
            d = json.loads(line)
            done.add((d["model"], d["dist"], d["seed"]))
    shard = int(os.environ.get("SHARD", "0"))
    nshards = int(os.environ.get("NSHARDS", "1"))
    only = os.environ.get("ONLY")  # e.g. "g11:t,g11:skewt"
    with path.open("a") as fh:
        for i in range(seeds):
            if i % nshards != shard:
                continue
            for model, (_, _, dists) in MODELS.items():
                for dist in dists:
                    if only and f"{model}:{dist}" not in only.split(","):
                        continue
                    if (model, dist, SEED0 + i) in done:
                        continue
                    d = compare(model, dist, SEED0 + i, n)
                    fh.write(json.dumps(d) + "\n")
                    fh.flush()
                    print(f"done {model} {dist} {SEED0 + i}", flush=True)


def summary(outdir: str, limit: int = 10_000) -> None:
    rows = []
    for p in Path(outdir).glob("cases_*.jsonl"):
        rows += [json.loads(x) for x in p.read_text().splitlines()]
    rows = [d for d in rows if d["seed"] < SEED0 + limit]
    for model, (spec, _, dists) in MODELS.items():
        for dist in dists:
            rs = sorted((d for d in rows if d["model"] == model and d["dist"] == dist), key=lambda d: d["seed"])
            if not rs:
                continue
            names = rs[0]["names"]
            print(f"\n{spec.name} {dist}: {len(rs)} seeds, n {rs[0]['n']}, arch converged "
                  f"{sum(d['arch_converged'] for d in rs)}, arch fit median "
                  f"{np.median([d['arch_seconds'] for d in rs]):.2f} s")
            B = np.array([d["batch"] for d in rs])
            print("  arch mean " + ", ".join(f"{nm} {B[:, j].mean():.3f}" for j, nm in enumerate(names)))
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
                    + ", ".join(f"{nm} {np.median(np.abs(G[:, j])):.2f}" for j, nm in enumerate(names))
                )
                print(
                    f"    shape gap to arch (abs): median "
                    + ", ".join(f"{names[nk + j]} {np.median(np.abs(SG[:, j])):.3f}" for j in range(SG.shape[1]))
                    + "; max " + ", ".join(f"{np.max(np.abs(SG[:, j])):.3f}" for j in range(SG.shape[1]))
                    + f"; arch shape SE median "
                    + ", ".join(f"{np.median([d['se'][nk + j] for d in rs]):.3f}" for j in range(SG.shape[1]))
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


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "one":
        n = int(sys.argv[5]) if len(sys.argv) > 5 else 10_000
        print(json.dumps(compare(sys.argv[2], sys.argv[3], int(sys.argv[4]), n), indent=1))
    elif cmd == "grid":
        grid(sys.argv[2], *(int(x) for x in sys.argv[3:]))
    elif cmd == "summary":
        summary(sys.argv[2], *(int(x) for x in sys.argv[3:]))
