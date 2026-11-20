"""Study C4 (review fable2): does stabilized forgetting (B) beat today's decaying
ridge (A) in `hmm` and `ew_class` without losing anywhere, and how does a
relative floor (C) compare?

    uv run python scripts/study_c4.py --out .claude/coord/review-fable2/c4-study
    uv run python scripts/study_c4.py --quick --out DIR      # one seed, fewer cells
    uv run python scripts/study_c4.py --cost --out DIR       # speed, both layouts

Each variant runs through `crates/rheon-core/examples/study_c4.rs` (built here
with the `study-c4` feature) on streams simulated from known parameters, and is
scored against the exact optimum computed from the true parameters:

- `hmm`: hmmlearn's forward algorithm (`hmmlearn._hmmc.forward_log`) with the
  true start probabilities and transition matrix, on scipy's Gaussian log
  densities at the true (possibly drifting) means and covariances: the
  optimal filtered posterior and the optimal row log-likelihood.
- `ew_class`: the Gaussian Bayes classifier at the true class priors, means
  and covariances (scipy's densities); with no decay, scikit-learn's QDA with
  `reg_param` where the definitions coincide.

Metrics per cell: the posterior's mean and largest absolute distance from the
optimum, the mean row log-likelihood less the optimum's (`hmm`), the rate of
argmax agreement with the optimum, and accuracy against the true state or
class. The variants (`crates/rheon-core/src/study_c4.rs` states the
recursions): A today; B stabilized forgetting with one pseudo-row of prior;
C a floor of 1/100 of the data scale.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from hmmlearn import _hmmc
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "target" / "release" / "examples" / "study_c4"
STAY = 0.95
SEP = 3.0
PRIOR = 1e-3
WARM = 200


def warm_rows(d: int) -> int:
    """The hmm's warm-up: 150 rows a feature, at least `WARM`. Shorter, the
    k-means seeding at 20 features leaves one state holding both regimes in
    steady data under every variant (200 rows: 2 of 3 seeds failed; 3,000: none)."""
    return max(WARM, 150 * d)


VARIANTS = {"A": "A", "B": "B:1", "C": "C:0.01"}


def variants(d: int) -> dict[str, tuple[str, float]]:
    """The main grid's variants: A; B at one pseudo-row and at `d` (scikit-learn's
    `BayesianGaussianMixture` puts its Wishart prior's degrees of freedom at
    the number of features); C at 1/100 of the data scale."""
    return {"A": ("A", PRIOR), "B": ("B:1", PRIOR), "Bd": (f"B:{d}", PRIOR), "C": ("C:0.01", PRIOR)}


def build(target: Path | None = None, rustflags: str | None = None) -> Path:
    env = dict(os.environ, CARGO_BUILD_JOBS=os.environ.get("CARGO_BUILD_JOBS", "6"))
    cmd = ["cargo", "build", "--release", "-p", "rheon-core", "--features", "study-c4"]
    cmd += ["--example", "study_c4"]
    if target is not None:
        env["CARGO_TARGET_DIR"] = str(target)
    if rustflags is not None:
        env["RUSTFLAGS"] = rustflags
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return (target or ROOT / "target") / "release" / "examples" / "study_c4"


# ---------------------------------------------------------------- the data


def true_params(rng: np.random.Generator, k: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """`k` Gaussians: covariances with eigenvalues in [1/2, 2] on random axes, the
    means `SEP` apart in the pooled covariance's Mahalanobis distance."""
    covs = []
    for _ in range(k):
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        eig = np.exp(rng.uniform(np.log(0.5), np.log(2.0), d))
        covs.append((q * eig) @ q.T)
    covs_a = np.array(covs)
    chol = np.linalg.cholesky(covs_a.mean(axis=0))
    means = [np.zeros(d)]
    for _ in range(1, k):
        u = rng.standard_normal(d)
        means.append(SEP * chol @ (u / np.linalg.norm(u)))
    return np.array(means), covs_a


def markov(rng: np.random.Generator, n: int, k: int, start: int | None = None) -> np.ndarray:
    z = np.empty(n, dtype=np.int64)
    z[0] = rng.integers(k) if start is None else start
    moves = rng.random(n) > STAY
    jumps = rng.integers(1, k, n)
    for t in range(1, n):
        z[t] = (z[t - 1] + jumps[t]) % k if moves[t] else z[t - 1]
    return z


def stream(spec: dict) -> dict:
    """A scenario's stream: clock steps, labels, features, and the true
    parameters row by row (`shift`, the means' drift, and `cscale`, the
    covariances' scale, both per row)."""
    rng = np.random.default_rng(spec["seed"])
    d, k, hl, model = spec["d"], 2, spec["hl"], spec["model"]
    means, covs = true_params(rng, k, d)
    kind = spec["kind"]
    hl_rows = hl if np.isfinite(hl) else 5000
    labels = (
        (lambda n, start=None: markov(rng, n, k, start))
        if model == "hmm"
        else (lambda n, start=None: rng.integers(0, k, n))
    )
    warm = warm_rows(d) if model == "hmm" else 0
    if kind in ("steady", "drift", "limit"):
        n = int(np.clip(20 * hl_rows, 5000, 100_000)) + warm
        z = labels(n)
        dt = np.ones(n)
        phase = np.zeros(n, dtype=np.int64)
    elif kind == "absent":
        n1 = int(np.clip(10 * hl_rows, 2000, 40_000)) + warm
        gap = spec["gap"] * hl
        n2 = int(min(gap, 20_000))
        n3 = 3000
        z = np.concatenate([labels(n1), np.zeros(n2, dtype=np.int64), labels(n3, 1)])
        if model != "hmm":
            z[n1 + n2] = 1
        dt = np.concatenate([np.ones(n1), np.full(n2, gap / n2), np.ones(n3)])
        phase = np.repeat([0, 1, 2], [n1, n2, n3])
        n = n1 + n2 + n3
    else:
        raise ValueError(kind)
    shift = np.zeros((n, k, d))
    cscale = np.ones(n)
    if kind == "drift":
        # The means walk together, 0.2 Mahalanobis units a half-life in a
        # fixed direction (their separation stays SEP), and the covariances'
        # scale swings between 1/2 and 2 with a period of 20 half-lives.
        t = np.arange(n) / hl_rows
        u = rng.standard_normal(d)
        u = np.linalg.cholesky(covs.mean(axis=0)) @ (u / np.linalg.norm(u))
        shift[:] = (0.2 * t)[:, None, None] * u[None, None, :]
        cscale = 2.0 ** np.sin(2 * np.pi * t / 20)
    chols = np.linalg.cholesky(covs)
    eps = rng.standard_normal((n, d))
    x = (
        means[z]
        + shift[np.arange(n), z]
        + np.sqrt(cscale)[:, None] * np.einsum("nij,nj->ni", chols[z], eps)
    )
    scale = spec.get("scale", 1.0)
    return dict(
        x=x * scale,
        z=z,
        dt=dt,
        phase=phase,
        means=means,
        covs=covs,
        shift=shift,
        cscale=cscale,
        scale=scale,
    )


def log_frames(st: dict) -> np.ndarray:
    """`ln N(x_t | μ_s(t), c_t·Σ_s)` per row and state, at the true parameters
    (scipy's density, on the unscaled row)."""
    x = st["x"] / st["scale"]
    n, d = x.shape
    out = np.empty((n, len(st["means"])))
    root = np.sqrt(st["cscale"])[:, None]
    for s, (m, c) in enumerate(zip(st["means"], st["covs"], strict=True)):
        dev = (x - m - st["shift"][:, s, :]) / root
        out[:, s] = multivariate_normal(mean=np.zeros(d), cov=c).logpdf(dev)
    out -= 0.5 * d * np.log(st["cscale"])[:, None]
    return out - d * np.log(st["scale"])


def hmm_oracle(st: dict) -> tuple[np.ndarray, np.ndarray]:
    """hmmlearn's forward pass at the true parameters: the filtered posterior
    and each row's log-likelihood `ln p(x_t | x_<t)`."""
    lf = log_frames(st)
    trans = np.array([[STAY, 1 - STAY], [1 - STAY, STAY]])
    _, fwd = _hmmc.forward_log(np.array([0.5, 0.5]), trans, lf)
    norm = logsumexp(fwd, axis=1)
    post = np.exp(fwd - norm[:, None])
    ll = np.diff(np.concatenate([[0.0], norm]))
    return post, ll


def class_oracle(st: dict) -> np.ndarray:
    """The Gaussian Bayes classifier at the true priors and parameters: 1/2 a
    class, and during an absence the present class alone."""
    lf = log_frames(st)
    prior = np.full(lf.shape, 0.5)
    prior[st["phase"] == 1] = [1.0, 0.0]
    with np.errstate(divide="ignore"):
        lp = lf + np.log(prior)
    return np.exp(lp - logsumexp(lp, axis=1)[:, None])


# ---------------------------------------------------------------- the runs


def run(exe: Path, spec: dict, st: dict, variant: str, prior: float, work: Path, reps: int = 1):
    d, k = spec["d"], 2
    tag = f"{os.getpid()}"
    inp, out = work / f"in-{tag}.bin", work / f"out-{tag}.bin"
    w = np.ones(len(st["z"]))
    rows = np.column_stack([st["dt"], w, st["z"].astype(float), st["x"]]).astype("<f8")
    rows.tofile(inp)
    model = "hmm" if spec["model"] == "hmm" else "ewclass"
    args = [str(exe), model, spec["cov"], variant, str(d), str(k), repr(float(spec["hl"]))]
    args += [repr(prior), str(warm_rows(d)), str(inp), str(out), str(reps)]
    res = subprocess.run(args, capture_output=True, text=True, check=True)
    info = json.loads(res.stdout)
    o = np.fromfile(out, dtype="<f8").reshape(-1, 1 + k)
    inp.unlink()
    out.unlink()
    return o[:, 0], o[:, 1:], info


def metrics(p, o, z, ll, oll, mask, ref) -> dict:
    """The distances on `mask`'s rows; an `hmm`'s states matched to the truth on
    `ref`'s rows (the labelling that agrees most)."""
    ok = ~np.isnan(p).any(axis=1)
    if ref is not None:
        r = ref & ok
        if r.any() and np.mean(p[r].argmax(1) == z[r]) < np.mean(p[r, ::-1].argmax(1) == z[r]):
            p = p[:, ::-1]
    m = mask & ok
    out: dict[str, float] = dict(rows=int(mask.sum()), nan=int((mask & ~ok).sum()))
    if not m.any():
        return out
    diff = np.abs(p[m] - o[m])
    out.update(
        mae=float(diff.mean()),
        max=float(diff.max()),
        agree=float(np.mean(p[m].argmax(1) == o[m].argmax(1))),
        acc=float(np.mean(p[m].argmax(1) == z[m])),
        oacc=float(np.mean(o[m].argmax(1) == z[m])),
    )
    ones = m & (z == 1)
    if ones.any():
        out["own"] = float(np.mean(p[ones].argmax(1) == 1))
        out["oown"] = float(np.mean(o[ones].argmax(1) == 1))
    if oll is not None:
        out["llgap"] = float(np.mean(ll[m] - oll[m]))
    return out


def cell(job: dict) -> list[dict]:
    exe, work, spec, variants = Path(job["exe"]), Path(job["work"]), job["spec"], job["variants"]
    st = stream(spec)
    if spec["model"] == "hmm":
        o, oll = hmm_oracle(st)
    else:
        o, oll = class_oracle(st), None
    n = len(st["z"])
    if spec["kind"] == "absent":
        mask = st["phase"] == 2
        first = mask & (np.cumsum(mask) <= 500)
        half = st["phase"] == 0
        half &= np.arange(n) >= np.flatnonzero(half)[-1] // 2
        ref = half
    else:
        start = warm_rows(spec["d"]) if spec["model"] == "hmm" else 0
        mask = np.arange(n) >= start + (n - start) // 2
        first = None
        ref = mask
    out = []
    for name, (v, prior) in variants.items():
        ll, p, info = run(exe, spec, st, v, prior, work)
        rec = dict(spec, variant=name, **{"info": info})
        hmm = spec["model"] == "hmm"
        # An hmm's states carry no labels: each window is scored under the
        # matching of states to regimes that agrees most on it, and `kept`
        # scores the return under the matching the phase before the absence
        # had (a swap of labels on the return shows there alone).
        rec["m"] = metrics(p, o, st["z"], ll, oll, mask, mask if hmm else None)
        if first is not None:
            rec["m500"] = metrics(p, o, st["z"], ll, oll, first, mask if hmm else None)
            rec["kept"] = metrics(p, o, st["z"], ll, oll, mask, ref if hmm else None)
        out.append(rec)
    return out


def jobs(quick: bool) -> list[dict]:
    seeds = [1] if quick else [1, 2, 3]
    ds = [2, 8] if quick else [2, 8, 13, 20]
    hls: list[float] = [50, 500] if quick else [50, 500, 5000]
    gaps = [5, 20, 100] if quick else [5, 20, 100, 500]
    specs: list[tuple[dict[str, Any], dict]] = []
    c: dict[str, Any]
    for model in ("hmm", "ewclass"):
        for cov in ("full", "diag"):
            for d in ds:
                for hl in hls:
                    for seed in seeds:
                        c = dict(model=model, cov=cov, d=d, hl=hl, seed=seed)
                        specs.append((dict(c, kind="steady"), variants(d)))
                        specs.append((dict(c, kind="drift"), variants(d)))
                        for g in gaps:
                            specs.append((dict(c, kind="absent", gap=g), variants(d)))
    # (d) scales: A at a fixed precision_prior, A* at one scaled with the data.
    for model in ("hmm", "ewclass"):
        for cov in ("full", "diag"):
            for scale in (1e-6, 1e-3, 1.0, 1e3, 1e6):
                for seed in seeds:
                    var = variants(8)
                    var["A*"] = ("A", PRIOR * scale * scale)
                    c = dict(model=model, cov=cov, d=8, hl=500, seed=seed, scale=scale)
                    specs.append((dict(c, kind="steady", table="scale"), var))
                    specs.append((dict(c, kind="absent", gap=20, table="scale"), var))
    # (e) the limits: no decay (infinite memory) and a memory of 5 rows.
    for model in ("hmm", "ewclass"):
        for cov in ("full", "diag"):
            for d in (2, 8, 20):
                for hl in (float("inf"), 5.0):
                    for seed in seeds:
                        c = dict(model=model, cov=cov, d=d, hl=hl, seed=seed, table="limit")
                        specs.append((dict(c, kind="limit"), variants(d)))
    # Sensitivity of B's prior rows and C's floor.
    sens = {
        "A": ("A", PRIOR),
        "B0.25": ("B:0.25", PRIOR),
        "B1": ("B:1", PRIOR),
        "B4": ("B:4", PRIOR),
        "C0.001": ("C:0.001", PRIOR),
        "C0.01": ("C:0.01", PRIOR),
        "C0.1": ("C:0.1", PRIOR),
        "D1": ("D:1", PRIOR),
        "D4": ("D:4", PRIOR),
    }
    for model in ("hmm", "ewclass"):
        for cov, hl in (("full", 500), ("full", 50), ("diag", 500)):
            for seed in seeds:
                c = dict(model=model, cov=cov, d=8, hl=hl, seed=seed, table="sens")
                specs.append((dict(c, kind="steady"), sens))
                specs.append((dict(c, kind="drift"), sens))
                specs.append((dict(c, kind="absent", gap=20), sens))
                specs.append((dict(c, kind="absent", gap=100), sens))
    return [dict(spec=s, variants=v) for s, v in specs]


def qda_limit(out: Path, seeds: list[int]) -> list[dict]:
    """(e) With no decay, B is QDA's `reg_param` shrinkage toward `diag(ŝ)` at
    `reg_param = P₀/(n + P₀)`: scikit-learn's QDA on the rows scaled by
    `1/√ŝ` (ŝ recomputed here, by the model's definition) against `ew_class`
    under B, and A against QDA at `reg_param = 0`, on held-out rows the model
    scores at weight 0. QDA divides by `n − 1` where the moments divide by `n`;
    that and the ridge A carries are the whole difference."""
    res = []
    exe = EXE
    for d in (2, 8, 20):
        for seed in seeds:
            rng = np.random.default_rng(100 + seed)
            means, covs = true_params(rng, 2, d)
            n, m = 4000, 2000
            z = np.tile([0, 1], (n + m) // 2)
            chols = np.linalg.cholesky(covs)
            x = means[z] + np.einsum("nij,nj->ni", chols[z], rng.standard_normal((n + m, d)))
            # ŝ as the model tracks it at no decay: the labelled class's squared
            # deviation from its means before the row, once it has weight.
            sums = np.zeros((2, d))
            cnt = np.zeros(2)
            e = []
            for t in range(n):
                c = z[t]
                if cnt[c] > 0:
                    e.append((x[t] - sums[c] / cnt[c]) ** 2)
                sums[c] += x[t]
                cnt[c] += 1
            shat = np.mean(e, axis=0)
            w = np.concatenate([np.ones(n), np.zeros(m)])
            rows = np.column_stack([np.ones(n + m), w, z.astype(float), x]).astype("<f8")
            inp = out / f"qda-in-{os.getpid()}.bin"
            outp = out / f"qda-out-{os.getpid()}.bin"
            rows.tofile(inp)
            got = {}
            for name, v in (("A", "A"), ("B", "B:1")):  # D's weights are not QDA's
                args = [str(exe), "ewclass", "full", v, str(d), "2", "inf", repr(PRIOR), "0"]
                subprocess.run(args + [str(inp), str(outp)], capture_output=True, check=True)
                got[name] = np.fromfile(outp, dtype="<f8").reshape(-1, 3)[n:, 1:]
            inp.unlink()
            outp.unlink()
            reg = 1.0 / (n / 2 + 1.0)
            qb = QuadraticDiscriminantAnalysis(reg_param=reg, priors=[0.5, 0.5])
            qb.fit(x[:n] / np.sqrt(shat), z[:n])
            pb = qb.predict_proba(x[n:] / np.sqrt(shat))
            qa = QuadraticDiscriminantAnalysis(reg_param=0.0, priors=[0.5, 0.5]).fit(x[:n], z[:n])
            pa = qa.predict_proba(x[n:])
            res.append(
                dict(
                    d=d,
                    seed=seed,
                    reg_param=reg,
                    b_vs_qda=float(np.abs(got["B"] - pb).max()),
                    a_vs_qda=float(np.abs(got["A"] - pa).max()),
                    b_vs_qda_mean=float(np.abs(got["B"] - pb).mean()),
                    a_vs_qda_mean=float(np.abs(got["A"] - pa).mean()),
                )
            )
    return res


def cost(out: Path) -> list[dict]:
    """ns a row, best of 7, per model, shape, width and variant, in the default
    layout and with every function aligned to 64 bytes (a second layout)."""
    exes = {
        "default": build(),
        "aligned": build(
            ROOT / "target" / "study-c4-aligned", "-C llvm-args=-align-all-functions=6"
        ),
    }
    res = []
    for model in ("hmm", "ewclass"):
        for cov in ("full", "diag"):
            for d in (2, 8, 13, 20):
                spec = dict(model=model, cov=cov, d=d, hl=500, seed=1, kind="steady")
                st = stream(spec)
                for layout, exe in exes.items():
                    for name, v in VARIANTS.items():
                        _, _, info = run(exe, spec, st, v, PRIOR, out, reps=7)
                        res.append(dict(spec, layout=layout, variant=name, **info))
    return res


# ---------------------------------------------------------------- the report


def _mean(v: list[float]) -> float:
    return float(np.mean(v)) if v else float("nan")


def _f(v: float, nd: int = 3) -> str:
    return "–" if v != v else f"{v:.{nd}f}"


def failed(m: dict) -> bool:
    """A run whose fit failed on the window: accuracy 0.2 or more below the
    optimum's (one state holding both regimes, or a class lost)."""
    return m.get("acc", 0.0) < m.get("oacc", 1.0) - 0.2


def report(out: Path) -> str:
    rs = json.loads((out / "results.json").read_text())
    names = ["A", "B", "Bd", "C"]
    lines: list[str] = []
    w = lines.append

    def group(rows, key):
        g: dict = {}
        for r in rows:
            g.setdefault(key(r), {}).setdefault(r["variant"], []).append(r)
        return g

    main = [r for r in rs if "table" not in r]
    for kind, title in (("steady", "(a) steady"), ("drift", "(c) drifting")):
        w(f"\n#### {title}: mean over 3 seeds\n")
        w(
            "| model | cov | d | hl | MAE A / B / Bd / C | max A / B / Bd / C "
            "| agree A / B / Bd / C | loglik gap A / B / Bd / C | failed fits A / B / Bd / C |"
        )
        w("|---|---|---|---|---|---|---|---|---|")
        g = group(
            [r for r in main if r["kind"] == kind],
            lambda r: (r["model"], r["cov"], r["d"], r["hl"]),
        )
        for k in sorted(g):
            v = g[k]
            cols = []
            for met, nd in (("mae", 4), ("max", 3), ("agree", 4), ("llgap", 3)):
                cols.append(
                    " / ".join(
                        _f(_mean([r["m"][met] for r in v[n] if met in r["m"]]), nd) for n in names
                    )
                )
            fails = " / ".join(str(sum(failed(r["m"]) for r in v[n])) for n in names)
            w(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | " + " | ".join(cols) + f" | {fails} |")

    w(
        "\n#### (b) a regime or class absent, then back: the 3,000 rows of the return, "
        "9 runs a cell (3 half-lives x 3 seeds)\n"
    )
    w(
        "| model | cov | d | absent (half-lives) | recovered A / B / Bd / C | MAE A / B / Bd / C "
        "| MAE first 500 A / B / Bd / C | loglik gap A / B / Bd / C | labels kept A / B / Bd / C |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    g = group(
        [r for r in main if r["kind"] == "absent"],
        lambda r: (r["model"], r["cov"], r["d"], r["gap"]),
    )
    for k in sorted(g):
        v = g[k]
        rec = " / ".join(f"{sum(not failed(r['m']) for r in v[n])}/{len(v[n])}" for n in names)
        mae = " / ".join(_f(_mean([r["m"]["mae"] for r in v[n]]), 3) for n in names)
        m5 = " / ".join(_f(_mean([r["m500"]["mae"] for r in v[n]]), 3) for n in names)
        ll = " / ".join(
            _f(_mean([r["m"]["llgap"] for r in v[n] if "llgap" in r["m"]]), 2) for n in names
        )
        kept = " / ".join(
            f"{sum(r['kept'].get('own', 0) > 0.5 and not failed(r['m']) for r in v[n])}"
            for n in names
        )
        w(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {rec} | {mae} | {m5} | {ll} | {kept} |")

    w(
        "\n#### (b) by half-life, `hmm` only: runs recovered, A / B / Bd / C "
        "(of 3 seeds x 4 widths)\n"
    )
    w("| cov | hl | absent 5 | absent 20 | absent 100 | absent 500 |")
    w("|---|---|---|---|---|---|")
    g = group(
        [r for r in main if r["kind"] == "absent" and r["model"] == "hmm"],
        lambda r: (r["cov"], r["hl"], r["gap"]),
    )
    for cov in ("full", "diag"):
        for hl in (50, 500, 5000):
            cells = []
            for gap in (5, 20, 100, 500):
                v = g.get((cov, hl, gap), {})
                cells.append(
                    " / ".join(str(sum(not failed(r["m"]) for r in v.get(n, []))) for n in names)
                )
            w(f"| {cov} | {hl} | " + " | ".join(cells) + " |")

    sc = [r for r in rs if r.get("table") == "scale"]
    w(
        "\n#### (d) data scale: d = 8, hl = 500, MAE (mean over 3 seeds); A* is A with "
        "precision_prior scaled by scale²\n"
    )
    w("| model | cov | scenario | scale | A | A* | B | Bd | C |")
    w("|---|---|---|---|---|---|---|---|---|")
    g = group(sc, lambda r: (r["model"], r["cov"], r["kind"], r["scale"]))
    for k in sorted(g):
        v = g[k]
        w(
            f"| {k[0]} | {k[1]} | {k[2]} | {k[3]:g} | "
            + " | ".join(
                _f(_mean([r["m"]["mae"] for r in v[n]]), 4) for n in ("A", "A*", "B", "Bd", "C")
            )
            + " |"
        )
    w("\nLargest change of a run's MAE across the five scales, per variant:\n")
    spread: dict = {}
    g2 = group(sc, lambda r: (r["model"], r["cov"], r["kind"], r["seed"]))
    for v in g2.values():
        for n, runs in v.items():
            maes = [r["m"]["mae"] for r in runs]
            spread[n] = max(spread.get(n, 0.0), max(maes) - min(maes))
    w("| " + " | ".join(spread) + " |")
    w("|" + "---|" * len(spread))
    w("| " + " | ".join(f"{spread[n]:.2e}" for n in spread) + " |")

    lim = [r for r in rs if r.get("table") == "limit"]
    w(
        "\n#### (e) the limits: no decay (infinite memory) and a half-life of 5 rows; "
        "mean over 3 seeds\n"
    )
    w(
        "| model | cov | d | hl | MAE A / B / Bd / C | agree A / B / Bd / C "
        "| failed fits A / B / Bd / C |"
    )
    w("|---|---|---|---|---|---|---|")
    g = group(lim, lambda r: (r["model"], r["cov"], r["d"], r["hl"]))
    for k in sorted(g):
        v = g[k]
        mae = " / ".join(_f(_mean([r["m"]["mae"] for r in v[n]]), 4) for n in names)
        ag = " / ".join(_f(_mean([r["m"]["agree"] for r in v[n]]), 4) for n in names)
        fails = " / ".join(str(sum(failed(r["m"]) for r in v[n])) for n in names)
        w(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {mae} | {ag} | {fails} |")
    if (out / "qda.json").exists():
        q = json.loads((out / "qda.json").read_text())
        w(
            "\nNo decay against scikit-learn's QDA (largest |posterior difference| on 2,000 "
            "held-out rows): B against `reg_param = 1/(n_c + 1)` on rows scaled by `1/√ŝ`, "
            "A against `reg_param = 0`.\n"
        )
        w("| d | B vs QDA(reg_param) | A vs QDA(0) |")
        w("|---|---|---|")
        for d in sorted({r["d"] for r in q}):
            rows = [r for r in q if r["d"] == d]
            w(
                f"| {d} | {max(r['b_vs_qda'] for r in rows):.1e} | "
                f"{max(r['a_vs_qda'] for r in rows):.1e} |"
            )

    se = [r for r in rs if r.get("table") == "sens"]
    w(
        "\n#### Sensitivity: B's prior rows, C's floor and D's (d = 8); MAE, "
        "and runs recovered of 3 on a return\n"
    )
    vs = ["A", "B0.25", "B1", "B4", "C0.001", "C0.01", "C0.1", "D1", "D4"]
    w("| model | cov | hl | scenario | " + " | ".join(vs) + " |")
    w("|---|---|---|---|" + "---|" * len(vs))
    g = group(se, lambda r: (r["model"], r["cov"], r["hl"], r["kind"], r.get("gap")))
    for k in sorted(g, key=str):
        v = g[k]
        name = k[3] if k[4] is None else f"absent {k[4]}"
        cells = []
        for n in vs:
            m = _mean([r["m"]["mae"] for r in v[n]])
            if k[3] == "absent":
                cells.append(f"{m:.3f} ({sum(not failed(r['m']) for r in v[n])})")
            else:
                cells.append(f"{m:.4f}")
        w(f"| {k[0]} | {k[1]} | {k[2]} | {name} | " + " | ".join(cells) + " |")

    if (out / "cost.json").exists():
        c = json.loads((out / "cost.json").read_text())
        w(
            "\n#### Cost: ns a row (best of 7, 10,000 rows, hl = 500), default layout / "
            "every function aligned to 64 bytes; state in doubles\n"
        )
        w("| model | cov | d | A | B | C | B / A | state A / B / C |")
        w("|---|---|---|---|---|---|---|---|")
        g = group(c, lambda r: (r["model"], r["cov"], r["d"]))
        for k in sorted(g):
            v = g[k]
            cn = ["A", "B", "C"]
            t = {n: {r["layout"]: r["ns_per_row"] for r in v[n]} for n in cn}

            def cell(n: str, t: dict = t) -> str:
                return f"{t[n]['default']:.0f} / {t[n]['aligned']:.0f}"

            ratio = " / ".join(f"{t['B'][lay] / t['A'][lay]:.2f}" for lay in ("default", "aligned"))
            st = " / ".join(str(v[n][0]["state_doubles"]) for n in cn)
            w(
                f"| {k[0]} | {k[1]} | {k[2]} | {cell('A')} | {cell('B')} | {cell('C')} | {ratio} "
                f"| {st} |"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.report:
        print(report(a.out))
        return
    if a.cost:
        (a.out / "cost.json").write_text(json.dumps(cost(a.out), indent=1))
        return
    exe = build()
    js = jobs(a.quick)
    for j in js:
        j["exe"], j["work"] = str(exe), str(a.out)
    results = []
    with ProcessPoolExecutor(a.workers) as pool:
        for i, r in enumerate(pool.map(cell, js, chunksize=1)):
            results.extend(r)
            if i % 50 == 0:
                print(f"{i}/{len(js)}", file=sys.stderr, flush=True)
    (a.out / "results.json").write_text(json.dumps(results, indent=1))
    seeds = [1] if a.quick else [1, 2, 3]
    (a.out / "qda.json").write_text(json.dumps(qda_limit(a.out, seeds), indent=1))


if __name__ == "__main__":
    main()
