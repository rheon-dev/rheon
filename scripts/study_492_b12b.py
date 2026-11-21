"""Study 492, B12 part 2: is GARCH(2,2)'s failure against arch the weak identification of beta1 and beta2?
(docs/records/STUDY-492-GARCH-FAMILY.md, part 7). Python and numpy only; arch is the oracle.

    study_492_b12b.py run TRUTH OUTDIR [--n N] SEED...   # one truth, seeds, appended to OUTDIR/TRUTH_nN.jsonl
    study_492_b12b.py summary OUTDIR [N]                 # counts within 2 SE, arch corr, path gap
    study_492_b12b.py freeze TRUTH OUTDIR SEED...        # PACF step from a start frozen to arch's fit; freeze-summary OUTDIR
    study_492_b12b.py diagnose TRUTH SEED [N]            # trajectory at 500/2000/10000 rows; the freeze tests

Matched persistence: every truth keeps alpha1 = alpha2 = 0.05, beta1 + beta2 = 0.87 (total persistence 0.97) and
omega = 0.03 (unconditional variance 1), and varies only the split of the beta lags:
    m_ref   beta = (0.45, 0.42)   the reference truth (c) of part 1; ratio beta2 / beta1 = 0.93
    m_eq    beta = (0.435, 0.435) separation zero
    m_sep   beta = (0.60, 0.27)   ratio 0.45
    m_xsep  beta = (0.75, 0.12)   ratio 0.16 (an extra, stronger separation)
Stationarity: P1 = a1 + b1, P2 = a2 + b2; 1 - P1 z - P2 z^2 has roots outside the unit circle iff P1 + P2 < 1,
P2 - P1 < 1 and |P2| < 1; all four have P1 + P2 = 0.97 and P2 - P1 < 0.
The comparison is `study_492_b11.compare_g22` unchanged (same rows, same arch fit, same two steps).
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_492 as S  # noqa: E402
import study_492_b11 as B  # noqa: E402

TRUTHS = {
    "m_ref": np.array([0.03, 0.05, 0.05, 0.45, 0.42]),
    "m_eq": np.array([0.03, 0.05, 0.05, 0.435, 0.435]),
    "m_sep": np.array([0.03, 0.05, 0.05, 0.60, 0.27]),
    "m_xsep": np.array([0.03, 0.05, 0.05, 0.75, 0.12]),
}
N = 10_000


def run(truth: str, outdir: str, seeds: list[int], n: int) -> None:
    B.G22_TRUTH = TRUTHS[truth]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{truth}_n{n}.jsonl").open("a") as fh:
        for s in seeds:
            t = time.perf_counter()
            d = B.compare_g22(s, n)
            d["truth"], d["seconds"] = TRUTHS[truth].tolist(), time.perf_counter() - t
            fh.write(json.dumps(d) + "\n")
            fh.flush()
            print(truth, n, s, f"{d['seconds']:.1f}s", flush=True)


class Frozen:
    """Mixin: after the 500-row batch start, replace the start's parameters by `freeze` (arch's full-sample fit)
    and replay the 500 rows under them; everything after is the online step unchanged."""

    freeze: np.ndarray | None = None
    _rows: np.ndarray | None = None

    def update(self, r: float) -> None:
        if self.theta is None and len(self.buffer) + 1 == self.warm:
            self._rows = np.asarray(self.buffer + [r])
        super().update(r)

    def _start(self) -> None:
        super()._start()
        rows = self._rows
        self.theta = self.freeze.copy()
        self.st = self.spec.init(S.backcast(rows * rows))
        self.n_seen, self.mean_r2 = 0, 0.0
        self.H = np.zeros((self.spec.k - 1, self.spec.k - 1))
        if isinstance(self, B.OnlinePacf):
            self.x = None  # replay in phi coordinates, then map H as in the unfrozen start
        for x in rows:
            self._row(float(x), step=False)
        if isinstance(self, B.OnlinePacf):
            self.x = B.phi_to_x(self.theta[1:])
            _, J = B.x_to_phi(self.x)
            self.H = J.T @ self.H @ J


class FrozenTarget(Frozen, S.Online):
    pass


class FrozenPacf(Frozen, B.OnlinePacf):
    pass


CHECK = (500, 1000, 2000, 5000, 10000)


def _ll(r: np.ndarray, th: np.ndarray, m: int) -> float:
    fp = S.filter_path(B.G22, th, r[:m], S.backcast(r[:m] * r[:m]))
    return -0.5 * float(np.sum(S.LN_2PI + np.log(fp) + r[:m] ** 2 / fp))


def trajectory(cls, r: np.ndarray, bp: np.ndarray, path_b: np.ndarray, freeze: np.ndarray | None) -> dict:
    est = cls(B.G22, target=True)
    if freeze is not None:
        est.freeze = freeze
    snaps, n = {}, len(r)
    for i, x in enumerate(r, 1):
        est.update(float(x))
        if i in CHECK and i <= n:
            th = est.theta.copy()
            pred = np.asarray(est.preds)
            lo = max(i - 1000, S.WARM)
            rel = np.abs(pred[lo:i] - path_b[lo:i]) / path_b[lo:i] if i > S.WARM else np.array([math.nan])
            snaps[i] = dict(theta=th, projected=est.projected, path_max=float(np.max(rel)), ll=_ll(r, th, i))
    return snaps


def diagnose(truth: str, seed: int, n: int) -> None:
    B.G22_TRUTH = TRUTHS[truth]
    spec = B.G22
    r = S.simulate(spec, TRUTHS[truth], n, seed)
    res = S.arch_fit(spec, r)
    bp, se = np.asarray(res.params, float), np.asarray(res.std_err, float)
    path_b = np.asarray(res.conditional_volatility, float) ** 2
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    print(f"truth {truth} {TRUTHS[truth]}, seed {seed}, n {n}")
    print(f"arch full-sample {bp}  SE {se}  persistence {bp[1:].sum():.3f}  ll {float(res.loglikelihood):.2f}")
    # arch on the same prefixes: the batch estimate an online fit at that row would be chasing
    for m in (500, 2000):
        rm = S.arch_fit(spec, r[:m])
        print(f"arch on first {m:>5} rows {np.asarray(rm.params, float)}  persistence {np.asarray(rm.params, float)[1:].sum():.3f}")
    # does arch's solution lie inside the PACF map's region (so the step can reach it)?
    phi_rt, _ = B.x_to_phi(B.phi_to_x(bp[1:]))
    print(f"arch phi round-trips through the PACF map: {np.allclose(phi_rt, bp[1:], atol=1e-9)} (max diff {np.max(np.abs(phi_rt - bp[1:])):.2e})")
    for name, cls, fcls in (("target step", S.Online, FrozenTarget), ("PACF step", B.OnlinePacf, FrozenPacf)):
        for lab, c, fr in (("start = 500-row batch fit", cls, None), ("start frozen to arch's full fit", fcls, bp)):
            sn = trajectory(c, r, bp, path_b, fr)
            print(f"-- {name}, {lab}")
            for m, d in sn.items():
                th = d["theta"]
                print(
                    f"   {m:>6} rows: theta {th}  gap (SE) {(th - bp) / se}  projected {d['projected']}  "
                    f"path max rel (window) {100 * d['path_max']:.1f}%  ll(first {m}) {d['ll']:.2f} vs arch-theta {_ll(r, bp, m):.2f}"
                )


def freeze_run(truth: str, outdir: str, seeds: list[int], n: int) -> None:
    """The PACF step from a start frozen to arch's full-sample fit, over seeds: does the gap close?"""
    B.G22_TRUTH = TRUTHS[truth]
    spec = B.G22
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"freeze_{truth}_n{n}.jsonl").open("a") as fh:
        for sd in seeds:
            r = S.simulate(spec, TRUTHS[truth], n, sd)
            res = S.arch_fit(spec, r)
            bp, se = np.asarray(res.params, float), np.asarray(res.std_err, float)
            path_b = np.asarray(res.conditional_volatility, float) ** 2
            est = FrozenPacf(spec, target=True)
            est.freeze = bp
            for x in r:
                est.update(float(x))
            th = est.theta
            gap = (th - bp) / se
            rel = np.abs(np.asarray(est.preds)[n - 1000 :] - path_b[n - 1000 :]) / path_b[n - 1000 :]
            fp = S.filter_path(spec, th, r, S.backcast(r * r))
            ll_gap = -0.5 * float(np.sum(S.LN_2PI + np.log(fp) + r * r / fp)) - float(res.loglikelihood)
            d = dict(seed=sd, n=n, gap_se=gap.tolist(), within2_all=bool(np.all(np.abs(gap) <= 2)),
                     within2_noomega=bool(np.all(np.abs(gap[1:]) <= 2)), path_max=float(np.max(rel)),
                     ll_gap=ll_gap, projected=est.projected)
            fh.write(json.dumps(d) + "\n")
            fh.flush()
            print(truth, "freeze", sd, flush=True)


def freeze_summary(outdir: str, n: int) -> None:
    for truth in TRUTHS:
        p = Path(outdir) / f"freeze_{truth}_n{n}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        print(
            f"frozen start, truth {truth}: {len(rows)} seeds; within 2 SE all {sum(d['within2_all'] for d in rows)}, "
            f"without omega {sum(d['within2_noomega'] for d in rows)}; path gap max over last 1000 median "
            f"{100 * np.median([d['path_max'] for d in rows]):.1f}% worst {100 * max(d['path_max'] for d in rows):.1f}%; "
            f"ll gap median {np.median([d['ll_gap'] for d in rows]):.2f} worst {min(d['ll_gap'] for d in rows):.1f}; "
            f"projected rows median {np.median([d['projected'] for d in rows]):.0f}"
        )


def summary(outdir: str, n: int) -> None:
    for truth, th in TRUTHS.items():
        p = Path(outdir) / f"{truth}_n{n}.jsonl"
        if not p.exists():
            continue
        rows = sorted((json.loads(x) for x in p.read_text().splitlines()), key=lambda d: d["seed"])
        se = np.array([d["se"] for d in rows])
        print(
            f"truth {truth} {th.tolist()}: {len(rows)} seeds, n {n}; arch converged "
            f"{sum(d['arch_converged'] for d in rows)}; arch corr(b1, b2) median "
            f"{np.median([d['arch_corr_b1b2'] for d in rows]):.3f} (weakest {max(d['arch_corr_b1b2'] for d in rows):.3f}); "
            f"arch SE b1 / b2 median {np.median(se[:, 3]):.3f} / {np.median(se[:, 4]):.3f}"
        )
        for tag in ("target", "pacf"):
            v = [d[tag] for d in rows]
            G = np.abs(np.array([x["gap_se"] for x in v]))
            pg = [x["path_max_rel_last1000"] for x in v]
            pm = [x["path_median_rel_last1000"] for x in v]
            print(
                f"  {tag}: within 2 SE all {sum(x['within2_all'] for x in v)}, without omega "
                f"{sum(x['within2_noomega'] for x in v)}; persistence {sum(abs(x['pers_gap_se']) <= 2 for x in v)}, "
                f"beta sum {sum(abs(x['bsum_gap_se']) <= 2 for x in v)}; b1 only {int((G[:, 3] <= 2).sum())}, "
                f"b2 only {int((G[:, 4] <= 2).sum())}; median |gap| (SE) "
                + ", ".join(f"{np.median(G[:, j]):.2f}" for j in range(5))
                + f"\n    path gap (max over last 1000): median {100 * np.median(pg):.1f}% worst {100 * np.max(pg):.1f}% "
                f"(seed {rows[int(np.argmax(pg))]['seed']}); path gap (median over last 1000) median {100 * np.median(pm):.2f}%; "
                f"ll gap median {np.median([x['ll_gap'] for x in v]):.2f} worst {np.min([x['ll_gap'] for x in v]):.1f}"
            )


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "summary":
        summary(a[1], int(a[2]) if len(a) > 2 else N)
    elif a[0] == "freeze-summary":
        freeze_summary(a[1], int(a[2]) if len(a) > 2 else N)
    elif a[0] == "freeze":
        freeze_run(a[1], a[2], [int(x) for x in a[3:]], N)
    elif a[0] == "diagnose":
        diagnose(a[1], int(a[2]), int(a[3]) if len(a) > 3 else N)
    else:
        rest = a[3:]
        n = N
        if rest and rest[0] == "--n":
            n, rest = int(rest[1]), rest[2:]
        run(a[1], a[2], [int(x) for x in rest], n)
