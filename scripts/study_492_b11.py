"""Study 492, B11 part 2: a stationary GARCH(2,2) step and a stabiliser for the t's nu step
(docs/records/STUDY-492-GARCH-FAMILY.md, part 5). Numpy only; arch is the oracle.

    study_492_b11.py check                      # Jacobian and score checks, root moduli (seconds)
    study_492_b11.py g22 OUTDIR [SEEDS] [N]     # GARCH(2,2) Gaussian, targeted: Euclidean phi step vs PACF step
    study_492_b11.py t OUTDIR [SEEDS] [N] [SEED...]   # GARCH(1,1) t, free: the nu forms
    study_492_b11.py summary OUTDIR

(3) GARCH(2,2) in partial-autocorrelation coordinates. The squared residual is an ARMA(2,2)
    e2_t = omega + P1 e2_{t-1} + P2 e2_{t-2} + u_t - b1 u_{t-1} - b2 u_{t-2},  P_i = a_i + b_i,
so covariance stationarity is: the roots of 1 - P1 z - P2 z^2 lie outside the unit circle. The
Barndorff-Nielsen-Schou / Jones (1987) map takes the open square (-1, 1)^2 onto exactly that region:
    P2 = r2,  P1 = r1 (1 - r2),  r_i = tanh(u_i),  u in R^2.
arch's nonnegativity (a_i, b_i >= 0) adds P_i >= a_i >= 0, i.e. r_i >= 0 (u_i >= 0), and b_i = P_i - a_i
>= 0, i.e. 0 <= a_i <= P_i. The stepped vector is x = (a1, a2, u1, u2); the targeted omega is
v (1 - P1 - P2) = v (1 - r1)(1 - r2). The projection is closed form: u_i <- max(u_i, 0), then
a_i <- clip(a_i, 0, P_i(u)); the persistence margin 1 - P1 - P2 = (1 - r1)(1 - r2) >= 1e-4 is kept by
lowering r1 alone. The roots of 1 - b1 z - b2 z^2 are then outside too (b >= 0, b1 + b2 <= P1 + P2 < 1).
Chain rule: phi = (a1, a2, b1, b2) = (a1, a2, P1 - a1, P2 - a2),
    dP1/du1 = (1 - r2)(1 - r1^2),  dP1/du2 = -r1 (1 - r2^2),  dP2/du2 = 1 - r2^2,
    score_x = J' score_phi,  info_x = J' info_phi J  (J = dphi/dx, 4 x 4).

(4) nu in s = log(nu - 2). nu = 2 + exp(s) cannot cross 2; score_s = (nu - 2) score_nu,
info_s = (nu - 2)^2 info_nu. Plus a slower gain kappa on the shape's component of the joint step and a cap
|ds| <= s_cap per row (a relative move of nu - 2 of at most e^s_cap - 1).
"""

# ruff: noqa: E501, E741, B905, SIM108
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_492 as S  # noqa: E402

G22 = S.MODELS["garch22"][0]
G22_TRUTH = S.MODELS["garch22"][1]
U_MAX = 20.0  # tanh(20) = 1 - 8e-18: the bound on u that float can still tell from 1


# ------------------------------------------------------------------ (3) the PACF map


def x_to_phi(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x = (a1, a2, u1, u2) -> phi = (a1, a2, b1, b2) and J = dphi/dx."""
    a1, a2, u1, u2 = x
    r1, r2 = math.tanh(u1), math.tanh(u2)
    P1, P2 = r1 * (1.0 - r2), r2
    dP1 = np.array([(1.0 - r2) * (1.0 - r1 * r1), -r1 * (1.0 - r2 * r2)])
    dP2 = np.array([0.0, 1.0 - r2 * r2])
    J = np.zeros((4, 4))
    J[0, 0] = J[1, 1] = 1.0
    J[2, 0], J[2, 2:] = -1.0, dP1
    J[3, 1], J[3, 2:] = -1.0, dP2
    return np.array([a1, a2, P1 - a1, P2 - a2]), J


def phi_to_x(phi: np.ndarray) -> np.ndarray:
    a1, a2, b1, b2 = phi
    P1, P2 = a1 + b1, a2 + b2
    r2 = min(max(P2, 0.0), math.tanh(U_MAX))
    r1 = min(max(P1 / (1.0 - r2), 0.0), math.tanh(U_MAX))
    return np.array([a1, a2, math.atanh(r1), math.atanh(r2)])


def project_x(x: np.ndarray) -> np.ndarray:
    a1, a2, u1, u2 = x
    u1 = min(max(u1, 0.0), U_MAX)
    u2 = min(max(u2, 0.0), U_MAX)
    r1, r2 = math.tanh(u1), math.tanh(u2)
    if (1.0 - r1) * (1.0 - r2) < S.MARGIN:  # persistence margin, by lowering r1
        r1 = 1.0 - S.MARGIN / (1.0 - r2)
        u1 = math.atanh(max(r1, 0.0))
        r1 = math.tanh(u1)
    P1, P2 = r1 * (1.0 - r2), r2
    return np.array([min(max(a1, 0.0), P1), min(max(a2, 0.0), P2), u1, u2])


def max_root_moduli(phi: np.ndarray) -> tuple[float, float]:
    """Largest modulus of the reciprocal roots (the z^2 - c1 z - c2 roots) of 1 - P(L) and 1 - beta(L):
    below 1 is stationary / invertible."""
    a1, a2, b1, b2 = phi
    p = np.max(np.abs(np.roots([1.0, -(a1 + b1), -(a2 + b2)])))
    b = np.max(np.abs(np.roots([1.0, -b1, -b2]))) if (b1 or b2) else 0.0
    return float(p), float(b)


class OnlinePacf(S.Online):
    """Study 485's targeted step for GARCH(2,2), taken in x = (a1, a2, u1, u2)."""

    x: np.ndarray | None = None
    max_mod_p: float = 0.0
    max_mod_b: float = 0.0

    def _start(self) -> None:
        super()._start()  # phi-coordinate batch start and H; rebuild H in x at the start's fit
        self.x = phi_to_x(self.theta[1:])
        _, J = x_to_phi(self.x)
        self.H = J.T @ self.H @ J

    def _row(self, r: float, step: bool) -> None:
        if self.x is None:  # the start's replay in phi coordinates (H is mapped once in _start)
            return super()._row(r, step)
        spec = self.spec
        s2, dl = spec.predict(self.theta, self.st)
        if step:
            self.preds.append(s2)
        r2 = r * r
        self.n_seen += 1
        self.mean_r2 += (r2 - self.mean_r2) / self.n_seen
        vv = self.mean_r2
        dfree = dl[1:] - vv * spec.c_vec() * dl[0]
        _, J = x_to_phi(self.x)
        dx = J.T @ dfree
        info = 0.5 * np.outer(dx, dx)
        resid = r2 / s2 - 1.0
        if step:
            self.n_steps += 1
            gamma = 1.0 / (self.n_steps + self.t0)
            self.H += gamma * (info - self.H)
            move = gamma * np.linalg.solve(self.H + 1e-12 * np.eye(4), 0.5 * resid * dx)
            prop = self.x + move
            self.x = project_x(prop)
            if not np.array_equal(self.x, prop):
                self.projected += 1
            phi, _ = x_to_phi(self.x)
            mp, mb = max_root_moduli(phi)
            self.max_mod_p = max(self.max_mod_p, mp)
            self.max_mod_b = max(self.max_mod_b, mb)
            self.theta = np.concatenate([[vv * (1.0 - float(spec.c_vec() @ phi))], phi])
        else:
            self.H += (info - self.H) / self.n_seen
        spec.push(self.st, r, s2, dl)


def compare_g22(seed: int, n: int) -> dict:
    spec = G22
    r = S.simulate(spec, G22_TRUTH, n, seed)
    res = S.arch_fit(spec, r)
    bp = np.asarray(res.params, dtype=float)
    se = np.asarray(res.std_err, dtype=float)
    cov = np.asarray(res.param_cov, dtype=float)
    path_b = np.asarray(res.conditional_volatility, dtype=float) ** 2
    start = S.backcast(r * r)
    # identified combinations: the persistence a1 + a2 + b1 + b2 and the beta sum, in arch's SE of each
    cp = np.array([0, 1, 1, 1, 1.0])
    cb = np.array([0, 0, 0, 1, 1.0])
    se_p, se_b = math.sqrt(cp @ cov @ cp), math.sqrt(cb @ cov @ cb)
    corr_b = cov[3, 4] / math.sqrt(cov[3, 3] * cov[4, 4])
    out = {
        "seed": seed,
        "n": n,
        "batch": bp.tolist(),
        "se": se.tolist(),
        "arch_corr_b1b2": corr_b,
        "arch_converged": int(res.convergence_flag) == 0,
        "ll_batch": float(res.loglikelihood),
    }
    for tag, cls in (("target", S.Online), ("pacf", OnlinePacf)):
        est = cls(spec, target=True)
        t = time.perf_counter()
        for x in r:
            est.update(float(x))
        secs = time.perf_counter() - t - est.start_seconds
        th = est.theta
        pred = np.asarray(est.preds)
        tail = slice(n - 1000, n)
        rel = np.abs(pred[tail] - path_b[tail]) / path_b[tail]
        fp = S.filter_path(spec, th, r, start)
        gap = (th - bp) / se
        mp, mb = max_root_moduli(th[1:])
        out[tag] = {
            "theta": th.tolist(),
            "gap_se": gap.tolist(),
            "within2_all": bool(np.all(np.abs(gap) <= 2.0)),
            "within2_noomega": bool(np.all(np.abs(gap[1:]) <= 2.0)),
            "pers_gap_se": float(cp @ (th - bp)) / se_p,
            "bsum_gap_se": float(cb @ (th - bp)) / se_b,
            "path_max_rel_last1000": float(np.max(rel)),
            "path_median_rel_last1000": float(np.median(rel)),
            "ll_gap": -0.5 * float(np.sum(S.LN_2PI + np.log(fp) + r * r / fp))
            - float(res.loglikelihood),
            "projected_rows": est.projected,
            "max_mod_p_final": mp,
            "max_mod_b_final": mb,
            "max_mod_p_run": getattr(est, "max_mod_p", None),
            "max_mod_b_run": getattr(est, "max_mod_b", None),
            "us_per_row": 1e6 * secs / (n - S.WARM),
        }
    return out


# --------------------------------------------------------------- (4) the nu forms


class OnlineTs(S.OnlineT):
    """GARCH-t free form; the shape stepped in s = log(nu - 2) with gain kappa and a per-row cap."""

    log_nu: bool = True
    kappa: float = 1.0
    s_cap: float = math.inf

    def _row(self, r: float, step: bool) -> None:
        spec = self.spec
        s2, dl = spec.predict(self.theta, self.st)
        if step:
            self.preds.append(s2)
        self.n_seen += 1
        self.mean_r2 += (r * r - self.mean_r2) / self.n_seen
        nu = float(self.shape[0])
        z = np.array([r / math.sqrt(s2)])
        w, Sc = S.scores("t", z, nu, 0.0)
        jac = (nu - 2.0) if self.log_nu else 1.0
        sc = np.concatenate([0.5 * float(w[0]) * dl, [float(Sc[0, 0]) * jac]])
        if step:
            self.n_steps += 1
            gamma = 1.0 / (self.n_steps + self.t0)
            self.H += gamma * (np.outer(sc, sc) - self.H)
            move = gamma * np.linalg.solve(self.H + 1e-12 * np.eye(sc.shape[0]), sc)
            nf = dl.shape[0]
            prop = self.theta + move[:nf]
            new = spec.project(prop, self.mean_r2)
            if self.metric and not np.array_equal(new, prop):
                new = self._metric_step(prop, move, nf)
                move = np.concatenate([new - self.theta, move[nf:] + self._shape_shift])
            if not np.array_equal(new, prop):
                self.projected += 1
            self.theta = new
            ds = self.kappa * float(move[nf])
            ds = min(max(ds, -self.s_cap), self.s_cap)
            if self.log_nu:
                nu_new = 2.0 + (nu - 2.0) * math.exp(ds)
            else:
                nu_new = nu + ds
            self.shape = S.clip_shape(np.array([nu_new]))
        else:
            self.H += (np.outer(sc, sc) - self.H) / self.n_seen
        spec.push(self.st, r, s2, dl)


T_FORMS = {  # tag: (log_nu, kappa, s_cap, metric)
    "nu_euclid": (False, 1.0, math.inf, False),  # part 2's free form
    "nu_metric": (False, 1.0, math.inf, True),  # part 3's metric projection
    "logs_euclid": (True, 1.0, math.inf, False),
    "logs_metric": (True, 1.0, math.inf, True),
    "logs_metric_k0.1": (True, 0.1, math.inf, True),
    "logs_metric_cap": (True, 1.0, 0.01, True),
}


def compare_tforms(seed: int, n: int) -> dict:
    spec, truth, _ = S.T_MODELS["g11"]
    r = S.simulate_t(spec, truth, "t", n, seed)
    res = S.arch_fit_t(spec, "t", r)
    bp = np.asarray(res.params, dtype=float)
    se = np.asarray(res.std_err, dtype=float)
    path_b = np.asarray(res.conditional_volatility, dtype=float) ** 2
    out = {
        "seed": seed,
        "n": n,
        "batch": bp.tolist(),
        "se": se.tolist(),
        "ll_batch": float(res.loglikelihood),
    }
    base = OnlineTs(spec, "t")
    base.log_nu = False
    for x in r[: S.WARM]:
        base.update(float(x))  # the batch start (in nu; the replay below rebuilds H per form)
    for tag, (log_nu, kappa, cap, metric) in T_FORMS.items():
        est = copy.deepcopy(base)
        est.log_nu, est.kappa, est.s_cap, est.metric = log_nu, kappa, cap, metric
        est.H = np.zeros((4, 4))
        est.st = spec.init(S.backcast(r[: S.WARM] ** 2))
        est.n_seen, est.mean_r2 = 0, 0.0
        for x_ in r[: S.WARM]:
            est._row(float(x_), step=False)
        est.preds = [math.nan] * S.WARM
        t = time.perf_counter()
        try:
            for x in r[S.WARM :]:
                est.update(float(x))
        except (ZeroDivisionError, OverflowError, ValueError, np.linalg.LinAlgError) as e:
            out[tag] = {"failed": repr(e), "row": len(est.preds)}
            continue
        secs = time.perf_counter() - t
        full = np.concatenate([est.theta, est.shape])
        pred = np.asarray(est.preds)
        tail = slice(n - 1000, n)
        rel = np.abs(pred[tail] - path_b[tail]) / path_b[tail]
        gap = (full - bp) / se
        out[tag] = {
            "x": full.tolist(),
            "gap_se": gap.tolist(),
            "within2_all": bool(np.all(np.abs(gap) <= 2.0)),
            "path_max_rel_last1000": float(np.max(rel)),
            "ll_gap": S.loglik(spec, "t", full, r) - float(res.loglikelihood),
            "projected_rows": est.projected,
            "us_per_row": 1e6 * secs / (n - S.WARM),
        }
    return out


# ------------------------------------------------------------------------ check


def check() -> None:
    rng = np.random.default_rng(492)
    worst = 0.0
    for _ in range(200):  # the Jacobian against central differences, and the projection's image
        x = np.array(
            [rng.uniform(0, 0.2), rng.uniform(0, 0.2), rng.uniform(0, 4), rng.uniform(0, 4)]
        )
        x = project_x(x)
        _, J = x_to_phi(x)
        num = np.empty((4, 4))
        for j in range(4):
            e = np.zeros(4)
            e[j] = 1e-6
            num[:, j] = (x_to_phi(x + e)[0] - x_to_phi(x - e)[0]) / 2e-6
        worst = max(worst, float(np.max(np.abs(num - J))))
        phi = x_to_phi(x)[0]
        assert np.all(phi >= -1e-15), phi
        mp, mb = max_root_moduli(phi)
        assert mp < 1.0 and mb < 1.0, (mp, mb)
        assert np.allclose(x_to_phi(phi_to_x(phi))[0], phi, atol=1e-10)
    print(
        f"PACF map: Jacobian vs central difference max abs {worst:.2e} over 200 points; all images stationary"
    )
    # the score in x: J' (targeted Gaussian score in phi) against a central difference of the log-likelihood
    r = S.simulate(G22, G22_TRUTH, 800, 7)
    b = S.backcast(r * r)
    v = float(np.mean(r * r))
    full = S.objective(G22, r, b)
    c = G22.c_vec()

    def ll_x(x: np.ndarray) -> tuple[float, np.ndarray]:
        phi, J = x_to_phi(x)
        th = np.concatenate([[v * (1.0 - float(c @ phi))], phi])
        ll, g, _ = full(th, True)
        Jt = np.vstack([-v * c, np.eye(4)])
        return ll, J.T @ (Jt.T @ g)

    x0 = phi_to_x(G22_TRUTH[1:])
    _, g = ll_x(x0)
    num = np.array([(ll_x(x0 + h)[0] - ll_x(x0 - h)[0]) / 2e-6 for h in 1e-6 * np.eye(4)])
    rel = float(np.max(np.abs(g - num) / np.maximum(np.abs(num), 1e-3)))
    print(f"score in x at the truth: max rel gap to central difference {rel:.2e}")
    assert rel < 1e-5, rel
    # the t shape score in s = log(nu - 2)
    z = S.draws("t", 400, 3)
    nu = 6.0
    sc = float(np.sum(S.scores("t", z, nu, 0.0)[1][:, 0])) * (nu - 2.0)
    h = 1e-6
    f = lambda s: float(np.sum(S.logf(z, 2.0 + math.exp(s), 0.0)))  # noqa: E731
    numt = (f(math.log(nu - 2.0) + h) - f(math.log(nu - 2.0) - h)) / (2 * h)
    print(f"t score in s = log(nu - 2): rel gap {abs(sc - numt) / abs(numt):.2e}")
    assert abs(sc - numt) / abs(numt) < 1e-6


# ----------------------------------------------------------------------- runner


def run(family: str, outdir: str, seeds: list[int], n: int) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fn = compare_g22 if family == "g22" else compare_tforms
    with (out / f"{family}.jsonl").open("a") as fh:
        for s in seeds:
            t = time.perf_counter()
            d = fn(s, n)
            d["seconds"] = time.perf_counter() - t
            fh.write(json.dumps(d) + "\n")
            fh.flush()
            print(family, s, f"{d['seconds']:.1f}s", flush=True)


def summary(outdir: str) -> None:
    p = Path(outdir) / "g22.jsonl"
    if p.exists():
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        print(
            f"GARCH(2,2) Gaussian targeted, {len(rows)} seeds, n {rows[0]['n']}; arch corr(b1, b2) median "
            f"{np.median([d['arch_corr_b1b2'] for d in rows]):.2f}"
        )
        for tag in ("target", "pacf"):
            v = [d[tag] for d in rows]
            G = np.abs(np.array([x["gap_se"] for x in v]))
            print(
                f"  {tag}: within 2 SE all {sum(x['within2_all'] for x in v)}, without omega "
                f"{sum(x['within2_noomega'] for x in v)}; persistence within 2 SE "
                f"{sum(abs(x['pers_gap_se']) <= 2 for x in v)}, beta sum within 2 SE {sum(abs(x['bsum_gap_se']) <= 2 for x in v)}; "
                f"median |gap| "
                + ", ".join(f"{np.median(G[:, j]):.2f}" for j in range(5))
                + f"; max |gap| b1 {np.max(G[:, 3]):.1f} b2 {np.max(G[:, 4]):.1f}"
            )
            print(
                f"    path max rel last 1000: median {np.median([x['path_max_rel_last1000'] for x in v]):.4f}, "
                f"max {np.max([x['path_max_rel_last1000'] for x in v]):.4f}; median-rel max over seeds "
                f"{np.max([x['path_median_rel_last1000'] for x in v]):.4f}; ll gap median "
                f"{np.median([x['ll_gap'] for x in v]):.2f} min {np.min([x['ll_gap'] for x in v]):.2f}; projected rows "
                f"{sum(x['projected_rows'] for x in v)}; us/row {np.median([x['us_per_row'] for x in v]):.1f}"
            )
            if tag == "pacf":
                print(
                    f"    max reciprocal-root modulus over every step: 1-P(L) {max(x['max_mod_p_run'] for x in v):.6f}, "
                    f"1-beta(L) {max(x['max_mod_b_run'] for x in v):.6f}"
                )
            else:
                print(
                    f"    final max reciprocal-root modulus: 1-P(L) {max(x['max_mod_p_final'] for x in v):.6f}, "
                    f"1-beta(L) {max(x['max_mod_b_final'] for x in v):.6f}"
                )
    p = Path(outdir) / "t.jsonl"
    if p.exists():
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        print(f"GARCH(1,1) t, free, {len(rows)} seeds, n {rows[0]['n']}")
        for tag in T_FORMS:
            v = [d[tag] for d in rows]
            ok = [x for x in v if "failed" not in x]
            print(
                f"  {tag}: within 2 SE {sum(x['within2_all'] for x in ok)} of {len(v)} (failed {len(v) - len(ok)}); "
                f"ll gap median {np.median([x['ll_gap'] for x in ok]):.2f} min {np.min([x['ll_gap'] for x in ok]):.1f}; "
                f"path max rel median {np.median([x['path_max_rel_last1000'] for x in ok]):.4f} max "
                f"{np.max([x['path_max_rel_last1000'] for x in ok]):.3f}; us/row {np.median([x['us_per_row'] for x in ok]):.1f}"
            )
            for d in rows:
                if d["seed"] in (492111, 492134) and "failed" not in d[tag]:
                    x = d[tag]
                    print(
                        f"    {d['seed']}: nu {x['x'][3]:.2f} (arch {d['batch'][3]:.2f}), omega {x['x'][0]:.3f} "
                        f"(arch {d['batch'][0]:.3f}), max |gap| {np.max(np.abs(x['gap_se'])):.1f} SE, ll gap {x['ll_gap']:.1f}"
                    )


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "check":
        check()
    elif a[0] == "summary":
        summary(a[1])
    else:
        base = S.SEED0 if a[0] == "g22" else S.T_SEED0
        nseeds = int(a[2]) if len(a) > 2 else 10
        n = int(a[3]) if len(a) > 3 else 10_000
        seeds = [int(s) for s in a[4:]] or [base + i for i in range(nseeds)]
        run(a[0], a[1], seeds, n)
