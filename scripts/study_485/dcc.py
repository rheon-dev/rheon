"""DCC(1,1) online, two series with GARCH(1,1) margins: the correlation
recursion with ``(a, b)`` learned one row at a time, in numpy; the batch fit
written from the definition and minimized by scipy; the fifty-seed grid
(docs/records/ORACLE-STUDY-485.md, part 6).

    python dcc.py OUT.jsonl            # the grid, resuming after the last case written
    python dcc.py OUT.jsonl --summary  # its table
    python dcc.py --one N SEED         # one case, every field

The model (Engle 2002), on standardized residuals ``z_t = r_t / sigma_t``::

    Q_t = (1 - a - b) * Qbar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}
    R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)
    l_t = -1/2 * (ln|R_t| + z_t' R_t^{-1} z_t - z_t' z_t)        (second stage)

With two series ``R_t`` has one free entry, ``rho_t = q12 / sqrt(q11 q22)``,
and the composite likelihood over pairs is the full likelihood.

**Online.** Stage one: each margin is the online GARCH(1,1) of ``garch.py``
with variance targeting (its part-3 variant that passed), and ``z_t`` uses the
margin's variance for row ``t`` computed before row ``t`` was read. Stage two,
per row once both margins emit: ``Qbar`` is the running mean of ``z z'`` over
the rows before this one (online targeting, the scan's form (b)); ``rho_t`` is
emitted before ``z_t`` is read; then one Fisher-scoring step on ``l_t``::

    dQ_t/da = z_{t-1} z_{t-1}' - Qbar + b * dQ_{t-1}/da
    dQ_t/db = Q_{t-1}          - Qbar + b * dQ_{t-1}/db
    g_t     = dl_t/drho * grad rho_t,     I_t = (1 + rho^2)/(1 - rho^2)^2 * grad rho grad rho'
    H_t     = H_{t-1} + (I_t - H_{t-1}) / (t + t0)
    (a, b)  = Proj((a, b) + H_t^{-1} g_t / (t + t0))

``Proj``: ``a, b >= 0``, ``a + b <= 1 - 1e-4`` (Euclidean). The first ``warm``
DCC rows build ``H`` with ``(a, b)`` held at ``(0.03, 0.85)``. State past the
margins (two GARCH states, 128 bytes each): ``Qbar`` and ``Q`` (3 + 3), two
``dQ`` (6), the last ``z`` (2), ``(a, b)``, ``H`` (3), the count: 20 floats.

**Batch** (the oracle): arch's GARCH(1,1) fit per margin (zero mean, normal),
``z`` from its in-sample conditional volatility, ``Qbar`` the sample mean of
``z z'``, ``Q_0 = Qbar``, and ``(a, b)`` maximizing the summed ``l_t`` by
``scipy.optimize.minimize(method="SLSQP")`` under ``a, b >= 0``,
``a + b <= 1 - 1e-4``. The recursion is ``scipy.signal.lfilter``. Standard
errors: the robust sandwich ``H^{-1} J'J H^{-1}`` of the second stage, ``H``
and the per-row scores ``J`` by statsmodels' numerical differentiation. It
ignores the first stage's estimation error, so it is an approximation, not a
library's own standard error: no library here fits DCC (arch 9.0.0 has no
multivariate model).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import garch  # noqa: E402

TRUTH = (0.05, 0.90)  # a, b
MARGIN = (0.02, 0.08, 0.90)  # omega, alpha, beta of each margin
RHO_BAR = 0.5  # the off-diagonal of Qbar
SEEDS = range(485000, 485050)
SIZES = (2000, 10000)
MARGIN_EPS = 1e-4
NAMES = ("a", "b")


def simulate(n: int, seed: int, burn: int = 1000) -> np.ndarray:
    """``n`` rows of two returns: GARCH(1,1) margins, DCC(1,1) correlation."""
    a, b = TRUTH
    w, al, be = MARGIN
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal((n + burn, 2))
    qbar = np.array([[1.0, RHO_BAR], [RHO_BAR, 1.0]])
    q = qbar.copy()
    s2 = np.full(2, w / (1.0 - al - be))
    prev_r2 = s2.copy()
    prev_z = np.zeros(2)
    out = np.empty((n + burn, 2))
    for t in range(n + burn):
        if t > 0:
            q = (1.0 - a - b) * qbar + a * np.outer(prev_z, prev_z) + b * q
        rho = q[0, 1] / math.sqrt(q[0, 0] * q[1, 1])
        z = np.array([eta[t, 0], rho * eta[t, 0] + math.sqrt(1.0 - rho * rho) * eta[t, 1]])
        s2 = w + al * prev_r2 + be * s2
        r = np.sqrt(s2) * z
        out[t] = r
        prev_r2 = r * r
        prev_z = z
    return out[burn:]


def rho_path(a: float, b: float, z: np.ndarray, qbar: np.ndarray) -> np.ndarray:
    """``rho_t`` for every row, ``Q_0 = Qbar``, by ``lfilter`` on each entry."""
    from scipy.signal import lfilter

    n = z.shape[0]
    out = {}
    for i, j in ((0, 0), (1, 1), (0, 1)):
        x = z[:, i] * z[:, j]
        u = np.empty(n)
        u[0] = qbar[i, j]
        u[1:] = (1.0 - a - b) * qbar[i, j] + a * x[:-1]
        out[(i, j)] = lfilter([1.0], [1.0, -b], u)
    return out[(0, 1)] / np.sqrt(out[(0, 0)] * out[(1, 1)])


def loglik_rows(rho: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The second-stage log-likelihood of each row, two series."""
    d = 1.0 - rho * rho
    s = z[:, 0] ** 2 + z[:, 1] ** 2
    p = z[:, 0] * z[:, 1]
    return -0.5 * (np.log(d) + (s - 2.0 * rho * p) / d - s)


def batch_fit(r: np.ndarray) -> dict[str, object]:
    """The oracle (module docstring)."""
    from scipy.optimize import minimize
    from statsmodels.tools.numdiff import approx_fprime, approx_hess

    z = np.empty_like(r)
    for i in range(2):
        _, res = garch.arch_fit(r[:, i])
        z[:, i] = r[:, i] / np.asarray(res.conditional_volatility)
    qbar = z.T @ z / z.shape[0]

    def rows(theta: np.ndarray) -> np.ndarray:
        return loglik_rows(rho_path(theta[0], theta[1], z, qbar), z)

    def neg(theta: np.ndarray) -> float:
        return -float(np.sum(rows(theta)))

    fit = minimize(
        neg,
        x0=np.array([0.03, 0.85]),
        method="SLSQP",
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        constraints=[{"type": "ineq", "fun": lambda t: 1.0 - MARGIN_EPS - t[0] - t[1]}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    theta = fit.x
    hess = approx_hess(theta, lambda t: float(np.sum(rows(t))))
    scores = approx_fprime(theta, rows)
    hinv = np.linalg.inv(-hess)
    cov = hinv @ (scores.T @ scores) @ hinv
    return {
        "theta": theta,
        "se": np.sqrt(np.diag(cov)),
        "se_classic": np.sqrt(np.diag(hinv)),
        "success": bool(fit.success),
        "z": z,
        "qbar": qbar,
        "llf": -float(fit.fun),
    }


def project(a: float, b: float) -> tuple[float, float]:
    a, b = max(a, 0.0), max(b, 0.0)
    cap = 1.0 - MARGIN_EPS
    if a + b > cap:
        shift = 0.5 * (a + b - cap)
        a, b = a - shift, b - shift
        if a < 0.0:
            a, b = 0.0, cap
        elif b < 0.0:
            a, b = cap, 0.0
    return a, b


@dataclass
class OnlineDcc:
    """Stage two, online. ``update(z)`` takes one row of standardized residuals."""

    a0: float = 0.03
    b0: float = 0.85
    warm: int = 100
    t0: float = 100.0

    theta: np.ndarray = field(init=False)
    qbar: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    q: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    dqa: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    dqb: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    prev_z: np.ndarray = field(default_factory=lambda: np.zeros(2))
    H: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    n_seen: int = 0
    n_steps: int = 0
    pred: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.theta = np.array([self.a0, self.b0])

    def update(self, z: np.ndarray) -> None:
        a, b = self.theta
        if self.n_seen == 0:
            # no earlier row: Q starts at the identity, its derivatives at zero
            self.q = np.eye(2)
        else:
            zz = np.outer(self.prev_z, self.prev_z)
            q_prev = self.q
            self.q = (1.0 - a - b) * self.qbar + a * zz + b * q_prev
            self.dqa = zz - self.qbar + b * self.dqa
            self.dqb = q_prev - self.qbar + b * self.dqb
        q11, q22, q12 = self.q[0, 0], self.q[1, 1], self.q[0, 1]
        s = math.sqrt(q11 * q22)
        rho = q12 / s
        self.pred.append(rho)
        # the row's score and information
        d = 1.0 - rho * rho
        z1, z2 = float(z[0]), float(z[1])
        p, ss = z1 * z2, z1 * z1 + z2 * z2
        dl = rho / d + (p * d - rho * (ss - 2.0 * rho * p)) / (d * d)
        grad = np.array(
            [
                dq[0, 1] / s - 0.5 * rho * (dq[0, 0] / q11 + dq[1, 1] / q22)
                for dq in (self.dqa, self.dqb)
            ]
        )
        info = (1.0 + rho * rho) / (d * d) * np.outer(grad, grad)
        self.n_seen += 1
        if self.n_seen <= self.warm:
            self.H += (info - self.H) / self.n_seen
        else:
            self.n_steps += 1
            gamma = 1.0 / (self.n_steps + self.t0)
            self.H += gamma * (info - self.H)
            step = gamma * np.linalg.solve(self.H, dl * grad)
            self.theta = np.array(project(*(self.theta + step)))
        self.qbar += (np.outer(z, z) - self.qbar) / self.n_seen
        self.prev_z = np.asarray(z, dtype=float).copy()


def online(r: np.ndarray, **kw: object) -> tuple[OnlineDcc, np.ndarray, np.ndarray]:
    """Both stages online. Returns the DCC state, the rows it saw (those where
    both margins emit) and the standardized residuals it was fed."""
    s2 = np.column_stack([np.asarray(garch.run(r[:, i], target="running").pred) for i in range(2)])
    rows = np.flatnonzero(np.all(np.isfinite(s2), axis=1))
    z = r[rows] / np.sqrt(s2[rows])
    est = OnlineDcc(**kw)  # type: ignore[arg-type]
    for row in z:
        est.update(row)
    return est, rows, z


def compare(n: int, seed: int) -> dict[str, float]:
    r = simulate(n, seed)
    bf = batch_fit(r)
    th_b, se = bf["theta"], bf["se"]
    est, rows, _ = online(r)
    th_o = est.theta
    z_b, qbar_b = bf["z"], bf["qbar"]
    rho_b = rho_path(th_b[0], th_b[1], z_b, qbar_b)
    rho_bo = rho_path(th_o[0], th_o[1], z_b, qbar_b)
    pred = np.full(n, np.nan)
    pred[rows] = est.pred
    tail = slice(n // 2, n)
    llf_o = float(np.sum(loglik_rows(rho_bo, z_b)))
    out: dict[str, float] = {
        "n": float(n),
        "seed": float(seed),
        "batch_success": float(bf["success"]),
        "llf_batch": bf["llf"],
        "llf_diff_per_row": (bf["llf"] - llf_o) / n,
        "rho_gap_median_second_half": float(np.median(np.abs(pred[tail] - rho_b[tail]))),
        "rho_gap_p90_second_half": float(np.percentile(np.abs(pred[tail] - rho_b[tail]), 90)),
        "first_dcc_row": float(rows[0]),
    }
    for i, name in enumerate(NAMES):
        out[f"{name}_batch"] = float(th_b[i])
        out[f"{name}_online"] = float(th_o[i])
        out[f"{name}_batch_se"] = float(se[i])
        out[f"{name}_batch_se_classic"] = float(bf["se_classic"][i])
        out[f"{name}_diff_in_se"] = float((th_o[i] - th_b[i]) / se[i])
    return out


def load(out: Path) -> list[dict]:
    if not out.exists():
        return []
    return [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]


def grid(out: Path) -> None:
    finished = {(int(r["seed"]), int(r["n"])) for r in load(out)}
    with out.open("a", encoding="utf-8") as f:
        for n in SIZES:
            for seed in SEEDS:
                if (seed, n) in finished:
                    continue
                f.write(json.dumps(compare(n, seed)) + "\n")
                f.flush()
                print(f"dcc: n {n} seed {seed} done", flush=True)


def summary(out: Path) -> None:
    recs = load(out)
    print("| rows | a: median / max / over 3 | b | all within 2 SE | rho gap: median / max |")
    print("|---|---|---|---|---|")
    for n in SIZES:
        rows = [r for r in recs if int(r["n"]) == n]
        cells = []
        for name in NAMES:
            a = np.abs([r[f"{name}_diff_in_se"] for r in rows])
            cells.append(f"{np.median(a):.3g} / {a.max():.3g} / {int((a > 3).sum())}")
        within2 = sum(all(abs(r[f"{k}_diff_in_se"]) <= 2 for k in NAMES) for r in rows)
        g = np.array([r["rho_gap_median_second_half"] for r in rows])
        print(
            f"| {n:,} | "
            + " | ".join(cells)
            + f" | {within2} | {np.median(g):.3g} / {g.max():.3g} |"
        )
        llf = np.array([r["llf_diff_per_row"] for r in rows])
        ok = sum(r["batch_success"] == 1 for r in rows)
        ratio = np.median([r["a_batch_se"] / r["a_batch_se_classic"] for r in rows])
        print(
            f"<!-- n {n}: seeds {len(rows)}, scipy success {ok}; llf gap per row median "
            f"{np.median(llf):.3g} max {llf.max():.3g} min {llf.min():.3g}; "
            f"robust/classic SE of a, median {ratio:.3f} -->"
        )


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        for k, v in compare(int(sys.argv[2]), int(sys.argv[3])).items():
            print(f"{k:34s} {v: .10g}")
    elif "--summary" in sys.argv[2:]:
        summary(Path(sys.argv[1]))
    else:
        grid(Path(sys.argv[1]))
