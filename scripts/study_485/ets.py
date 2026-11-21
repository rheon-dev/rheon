"""ETS(A,A,N) online: the additive level-and-trend smoother with its smoothing
parameters learned one row at a time by a recursive prediction-error step, in
numpy; the comparison against statsmodels' ``ETSModel`` and the fifty-seed
grid (docs/records/ORACLE-STUDY-485.md, part 5).

    python ets.py OUT.jsonl            # the grid, resuming after the last case written
    python ets.py OUT.jsonl --summary  # its table
    python ets.py --one N SEED         # one case, every field

The model, in statsmodels' error-correction form (``ETSModel``'s ``beta``,
not the classical ``beta* = beta/alpha``)::

    yhat_t = l_{t-1} + b_{t-1},          e_t = y_t - yhat_t
    l_t    = l_{t-1} + b_{t-1} + alpha * e_t
    b_t    = b_{t-1} + beta * e_t

``yhat_t`` is emitted before ``y_t`` is read. The sensitivities of the states
to ``theta = (alpha, beta)`` are carried by their own recursion at the current
parameters, ``psi_t = d yhat_t / d theta = Sl_{t-1} + Sb_{t-1}``::

    Sl_t = (1 - alpha) * psi_t + (e_t, 0)
    Sb_t = Sb_{t-1} - beta * psi_t + (0, e_t)

and each row takes one Gauss-Newton step on ``e_t^2``::

    R_t     = R_{t-1} + (psi_t psi_t' - R_{t-1}) / t
    theta_t = Proj(theta_{t-1} + R_t^{-1} psi_t e_t / t)
    s2_t    = s2_{t-1} + (e_t^2 - s2_{t-1}) / t

``Proj`` is statsmodels' own admissible box: ``alpha`` in
``[1e-4, 1 - 1e-4]`` and ``beta / alpha`` in ``[1e-4, 1 - 1e-4]`` (so
``0 < beta < alpha``), applied after every step. The start: the initial level
and trend are given (as statsmodels' ``initialization_method="known"``),
``theta = (0.5, 0.05)``, and the first ``warm`` rows accumulate ``R`` with
``theta`` held. State: ``l, b`` (2), ``theta`` (2), ``Sl, Sb`` (4), ``R``
(3 distinct), ``s2`` and the row count: 13 floats, 104 bytes.

numpy only in the estimator; statsmodels is the oracle, imported only by the
comparison and the tests.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TRUTH = (0.3, 0.1, 1.0)  # alpha, beta, sigma
L0, B0 = 10.0, 0.1  # the initial level and trend, given to both sides
SEEDS = range(485000, 485050)
SIZES = (2000, 10000)
BOUND = 1e-4  # statsmodels' default margin on alpha and beta/alpha
NAMES = ("alpha", "beta", "sigma2")


def simulate(
    n: int, alpha: float, beta: float, sigma: float, seed: int, nu: float | None = None
) -> np.ndarray:
    """``n`` rows of ETS(A,A,N) from the given initial states. A local linear
    trend has no stationary distribution, so there is no burn-in. With ``nu``,
    the errors are Student-t with ``nu`` degrees of freedom scaled to standard
    deviation ``sigma``."""
    rng = np.random.default_rng(seed)
    e = sigma * (
        rng.standard_normal(n) if nu is None else rng.standard_t(nu, n) * math.sqrt((nu - 2.0) / nu)
    )
    y = np.empty(n)
    lev, tr = L0, B0
    for t in range(n):
        y[t] = lev + tr + e[t]
        lev, tr = lev + tr + alpha * e[t], tr + beta * e[t]
    return y


def one_step(alpha: float, beta: float, y: np.ndarray, l0: float = L0, b0: float = B0):
    """The fixed-parameter smoother: the one-step predictions."""
    out = np.empty(y.shape[0])
    lev, tr = l0, b0
    for t in range(y.shape[0]):
        out[t] = lev + tr
        e = float(y[t]) - out[t]
        lev, tr = lev + tr + alpha * e, tr + beta * e
    return out


def project(alpha: float, beta: float) -> tuple[float, float]:
    alpha = min(max(alpha, BOUND), 1.0 - BOUND)
    beta = min(max(beta, BOUND * alpha), (1.0 - BOUND) * alpha)
    return alpha, beta


@dataclass
class OnlineEts:
    """The recursive prediction-error estimator. ``update(y)`` takes one row."""

    l0: float = L0
    b0: float = B0
    alpha0: float = 0.5
    beta0: float = 0.05
    warm: int = 50

    lev: float = field(init=False)
    tr: float = field(init=False)
    theta: np.ndarray = field(init=False)
    Sl: np.ndarray = field(default_factory=lambda: np.zeros(2))
    Sb: np.ndarray = field(default_factory=lambda: np.zeros(2))
    R: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    s2: float = 0.0
    n_seen: int = 0
    pred: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.lev, self.tr = self.l0, self.b0
        self.theta = np.array([self.alpha0, self.beta0])

    def update(self, y: float) -> None:
        yhat = self.lev + self.tr
        self.pred.append(yhat)
        e = y - yhat
        psi = self.Sl + self.Sb
        self.n_seen += 1
        gamma = 1.0 / self.n_seen
        self.R += gamma * (np.outer(psi, psi) - self.R)
        self.s2 += gamma * (e * e - self.s2)
        alpha, beta = self.theta
        if self.n_seen > self.warm:
            a, b = self.theta + gamma * np.linalg.solve(self.R, psi * e)
            self.theta = np.array(project(float(a), float(b)))
        # the states and their sensitivities advance at the parameters the
        # row was predicted with
        self.Sl = (1.0 - alpha) * psi + np.array([e, 0.0])
        self.Sb = self.Sb - beta * psi + np.array([0.0, e])
        self.lev, self.tr = self.lev + self.tr + alpha * e, self.tr + beta * e

    @property
    def params(self) -> np.ndarray:
        return np.array([self.theta[0], self.theta[1], self.s2])


def run(y: np.ndarray, **kw: object) -> OnlineEts:
    est = OnlineEts(**kw)  # type: ignore[arg-type]
    for v in y:
        est.update(float(v))
    return est


def ets_fit(y: np.ndarray, l0: float = L0, b0: float = B0):  # noqa: ANN201
    """The oracle: ``ETSModel(error="add", trend="add", damped_trend=False)``
    with the initial states known, fitted with its defaults (L-BFGS on the
    likelihood with ``sigma2`` concentrated out, numerical gradient,
    ``beta/alpha`` bounded internally, ``cov_type="approx"``)."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    model = ETSModel(
        y,
        error="add",
        trend="add",
        damped_trend=False,
        initialization_method="known",
        initial_level=l0,
        initial_trend=b0,
    )
    return model, model.fit(disp=False)


def compare(
    n: int, seed: int, truth: tuple[float, float, float] = TRUTH, nu: float | None = None
) -> dict[str, float]:
    """The online estimator against the batch fit on one simulated series.
    ``alpha`` and ``beta`` gaps are in statsmodels' own standard errors;
    statsmodels gives none for ``sigma2`` (it is concentrated out), so its gap
    is in the Gaussian closed form ``sigma2 * sqrt(2/n)``. ``truth`` and ``nu``
    (Student-t errors) default to part 5's Gaussian series."""
    y = simulate(n, *truth, seed=seed, nu=nu)
    _, res = ets_fit(y)
    ab = np.asarray(res.params, dtype=float)
    ab_se = np.asarray(res.bse, dtype=float)
    yhat_b = one_step(ab[0], ab[1], y)
    e_b = y - yhat_b
    s2_b = float(np.mean(e_b * e_b))
    b = np.array([ab[0], ab[1], s2_b])
    se = np.array([ab_se[0], ab_se[1], s2_b * math.sqrt(2.0 / n)])
    est = run(y)
    o = est.params
    pred = np.asarray(est.pred)
    tail = slice(n // 2, n)
    out: dict[str, float] = {
        "n": float(n),
        "seed": float(seed),
        "sm_converged": float(bool(res.mle_retvals.get("converged", False))),
        "sm_fittedvalues_vs_ours_max": float(np.max(np.abs(res.fittedvalues - yhat_b))),
        "sm_mse_vs_ours": float(abs(res.mse - s2_b)),
        "online_mse_second_half": float(np.mean((y[tail] - pred[tail]) ** 2)),
        "batch_mse_second_half": float(np.mean(e_b[tail] ** 2)),
        "pred_rms_gap_second_half": float(np.sqrt(np.mean((pred[tail] - yhat_b[tail]) ** 2))),
    }
    for i, name in enumerate(NAMES):
        out[f"{name}_batch"] = float(b[i])
        out[f"{name}_online"] = float(o[i])
        out[f"{name}_batch_se"] = float(se[i])
        out[f"{name}_diff_in_se"] = float((o[i] - b[i]) / se[i])
    return out


def load(out: Path) -> list[dict]:
    if not out.exists():
        return []
    return [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]


def grid(
    out: Path,
    sizes: tuple[int, ...] = SIZES,
    truth: tuple[float, float, float] = TRUTH,
    nu: float | None = None,
) -> None:
    finished = {(int(r["seed"]), int(r["n"])) for r in load(out)}
    with out.open("a", encoding="utf-8") as f:
        for n in sizes:
            for seed in SEEDS:
                if (seed, n) in finished:
                    continue
                rec = compare(n, seed, truth, nu)
                rec.update(alpha_true=truth[0], beta_true=truth[1], nu=nu or -1.0)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"ets: n {n} seed {seed} done", flush=True)


def summary(out: Path) -> None:
    recs = load(out)
    print("| rows | alpha: median / max / over 3 | beta | sigma2 | all within 2 SE |")
    print("|---|---|---|---|---|")
    for n in sorted({int(r["n"]) for r in recs}):
        rows = [r for r in recs if int(r["n"]) == n]
        cells = []
        for name in NAMES:
            a = np.abs([r[f"{name}_diff_in_se"] for r in rows])
            cells.append(f"{np.median(a):.3g} / {a.max():.3g} / {int((a > 3).sum())}")
        within2 = sum(all(abs(r[f"{k}_diff_in_se"]) <= 2 for k in NAMES) for r in rows)
        within2_ab = sum(all(abs(r[f"{k}_diff_in_se"]) <= 2 for k in NAMES[:2]) for r in rows)
        print(f"| {n:,} | " + " | ".join(cells) + f" | {within2} |")
        gap = np.array([r["pred_rms_gap_second_half"] for r in rows])
        ratio = np.array([r["online_mse_second_half"] / r["batch_mse_second_half"] for r in rows])
        conv = sum(r["sm_converged"] == 1 for r in rows)
        fv = max(r["sm_fittedvalues_vs_ours_max"] for r in rows)
        mse = max(r["sm_mse_vs_ours"] for r in rows)
        at_bound = sum(r["beta_online"] >= (1 - BOUND) * r["alpha_online"] * 0.999 for r in rows)
        print(
            f"<!-- n {n}: seeds {len(rows)}, converged {conv}; alpha and beta within 2: "
            f"{within2_ab}; pred rms gap median {np.median(gap):.3g} max {gap.max():.3g}; "
            f"mse ratio median {np.median(ratio):.4f} max {ratio.max():.4f}; "
            f"fittedvalues max gap {fv:.2g}; mse gap {mse:.2g}; "
            f"online beta at its bound {at_bound} -->"
        )
        # the per-seed bounds proposed in the record's section 15.2
        se_ok = sum(all(abs(r[f"{k}_diff_in_se"]) <= 2.5 for k in NAMES) for r in rows)
        rms_ok = sum(r["pred_rms_gap_second_half"] <= 0.05 for r in rows)
        both = sum(
            all(abs(r[f"{k}_diff_in_se"]) <= 2.5 for k in NAMES)
            and r["pred_rms_gap_second_half"] <= 0.05
            for r in rows
        )
        worst = max(rows, key=lambda r: max(abs(r[f"{k}_diff_in_se"]) for k in NAMES))
        wk = max(NAMES, key=lambda k: abs(worst[f"{k}_diff_in_se"]))
        print(
            f"<!-- n {n}: section 15.2 bounds: all within 2.5 SE {se_ok}; rms gap <= 0.05 "
            f"{rms_ok}; both {both}; worst seed {int(worst['seed'])} {wk} "
            f"{worst[f'{wk}_diff_in_se']:+.3g} SE -->"
        )


def _opt(name: str) -> float | None:
    """``--name VALUE`` from the command line, or None."""
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else None


if __name__ == "__main__":
    # options (request A10, part 3): --alpha A sets the truth to (A, A/3, 1.0),
    # keeping part 5's beta/alpha of 1/3; --nu NU gives Student-t errors;
    # --rows N runs that one row count
    alpha_opt, nu_opt, rows_opt = _opt("--alpha"), _opt("--nu"), _opt("--rows")
    truth_opt = TRUTH if alpha_opt is None else (alpha_opt, alpha_opt / 3.0, 1.0)
    sizes_opt = SIZES if rows_opt is None else (int(rows_opt),)
    if sys.argv[1] == "--one":
        for k, v in compare(int(sys.argv[2]), int(sys.argv[3]), truth_opt, nu_opt).items():
            print(f"{k:34s} {v: .10g}")
    elif "--summary" in sys.argv[2:]:
        summary(Path(sys.argv[1]))
    else:
        grid(Path(sys.argv[1]), sizes_opt, truth_opt, nu_opt)
