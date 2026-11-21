"""Study C4, Bd's fast form: `hmm` under stabilized forgetting as
`Hmm::with_stabilized` builds it (`F`, its factor form topping the prior up
lazily from 13 features) against today's ridge (`A`), the study's prototype
(`Bd`, which read the states before the row's decay) and the same recursion
read through co-moments at every width (`E`, exact; F is E below 13
features). Streams, oracles and metrics are `study_c4.py`'s.

    uv run python scripts/study_c4_bd.py --out DIR [--quick] [--workers N]
    uv run python scripts/study_c4_bd.py --out DIR --report
    uv run python scripts/study_c4_bd.py --out DIR --warm      # the warm-up rule
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_c4 as base  # noqa: E402

PRIOR = base.PRIOR


def variants(d: int, extra: dict[str, str] | None = None) -> dict[str, tuple[str, float]]:
    """A, the prototype's Bd, F and (from 13 features) E; or the extras alone."""
    if extra:
        return {name: (spec.format(d=d), PRIOR) for name, spec in extra.items()}
    v = {"A": ("A", PRIOR), "Bd": (f"B:{d}", PRIOR), "F": (f"F:{d}", PRIOR)}
    if d >= 13:
        v["E"] = (f"E:{d}", PRIOR)
    for name, spec in (extra or {}).items():
        v[name] = (spec.format(d=d), PRIOR)
    return v


def jobs(quick: bool, extra: dict[str, str] | None) -> list[dict]:
    seeds = [1] if quick else [1, 2, 3]
    ds = [2, 8, 13, 20]
    hls: list[float] = [50, 500] if quick else [50, 500, 5000]
    gaps = [5, 20, 100, 500]
    specs: list[tuple[dict[str, Any], dict]] = []
    for cov in ("full", "diag"):
        for d in ds:
            for hl in hls:
                for seed in seeds:
                    c = dict(model="hmm", cov=cov, d=d, hl=hl, seed=seed)
                    var = variants(d, extra)
                    specs.append((dict(c, kind="steady"), var))
                    specs.append((dict(c, kind="drift"), var))
                    for g in gaps:
                        specs.append((dict(c, kind="absent", gap=g), var))
    return [dict(spec=s, variants=v) for s, v in specs]


def warm_jobs() -> list[dict]:
    """Steady streams at 13, 20 and 32 features, hl = 500, under warm-ups of
    200 rows and of 10, 25, 50, 100 and 150 rows a feature: which fail."""
    specs = []
    for cov in ("full", "diag"):
        for d in (8, 13, 20, 32):
            for seed in (1, 2, 3, 4, 5, 6):
                for warm in (200, 10 * d, 25 * d, 50 * d, 100 * d, 150 * d):
                    c = dict(model="hmm", cov=cov, d=d, hl=500, seed=seed, warm=warm)
                    specs.append(
                        (
                            dict(c, kind="steady"),
                            {"A": ("A", PRIOR), "F": (f"F:{d}", PRIOR)},
                        )
                    )
    return [dict(spec=s, variants=v) for s, v in specs]


def warm_cell(job: dict) -> list[dict]:
    warm = job["spec"]["warm"]

    def rows(d: int) -> int:
        return int(warm)

    base.warm_rows = rows
    return base.cell(job)


def loss(a: list[dict], b: list[dict]) -> bool:
    """The study's loss: a mean MAE 2% and 5e-4 above A's, a log-likelihood
    0.01 nats below, or more failed fits."""
    ma = np.mean([r["m"].get("mae", np.nan) for r in a])
    mb = np.mean([r["m"].get("mae", np.nan) for r in b])
    la = np.mean([r["m"].get("llgap", np.nan) for r in a])
    lb = np.mean([r["m"].get("llgap", np.nan) for r in b])
    fa = sum(base.failed(r["m"]) for r in a)
    fb = sum(base.failed(r["m"]) for r in b)
    return bool(mb > ma * 1.02 and mb > ma + 5e-4 or lb < la - 0.01 or fb > fa)


def report(out: Path) -> str:
    rs = json.loads((out / "results-bd.json").read_text())
    names = [
        n for n in ("A", "Bd", "F", "E", "F4", "M1", "Md") if any(r["variant"] == n for r in rs)
    ]
    lines: list[str] = []
    w = lines.append

    def group(rows, key):
        g: dict = {}
        for r in rows:
            g.setdefault(key(r), {}).setdefault(r["variant"], []).append(r)
        return g

    hdr = " / ".join(names)
    for kind, title in (("steady", "(a) steady"), ("drift", "(c) drifting")):
        w(f"\n#### {title}: mean over the seeds\n")
        w(f"| cov | d | hl | MAE {hdr} | loglik gap {hdr} | failed fits {hdr} |")
        w("|---|---|---|---|---|---|")
        g = group([r for r in rs if r["kind"] == kind], lambda r: (r["cov"], r["d"], r["hl"]))
        for k in sorted(g):
            v = g[k]
            mae = " / ".join(
                base._f(base._mean([r["m"]["mae"] for r in v.get(n, [])]), 4) for n in names
            )
            ll = " / ".join(
                base._f(base._mean([r["m"]["llgap"] for r in v.get(n, []) if "llgap" in r["m"]]), 3)
                for n in names
            )
            fails = " / ".join(str(sum(base.failed(r["m"]) for r in v.get(n, []))) for n in names)
            w(f"| {k[0]} | {k[1]} | {k[2]} | {mae} | {ll} | {fails} |")

    w("\n#### (b) absent, then back: the return's 3,000 rows, runs a cell = half-lives x seeds\n")
    w(
        f"| cov | d | absent | recovered {hdr} | MAE {hdr} "
        f"| MAE first 500 {hdr} | loglik gap {hdr} |"
    )
    w("|---|---|---|---|---|---|---|")
    g = group([r for r in rs if r["kind"] == "absent"], lambda r: (r["cov"], r["d"], r["gap"]))
    for k in sorted(g):
        v = g[k]
        rec = " / ".join(
            f"{sum(not base.failed(r['m']) for r in v.get(n, []))}/{len(v.get(n, []))}"
            for n in names
        )
        mae = " / ".join(
            base._f(base._mean([r["m"]["mae"] for r in v.get(n, [])]), 3) for n in names
        )
        m5 = " / ".join(
            base._f(base._mean([r["m500"]["mae"] for r in v.get(n, [])]), 3) for n in names
        )
        ll = " / ".join(
            base._f(base._mean([r["m"]["llgap"] for r in v.get(n, []) if "llgap" in r["m"]]), 2)
            for n in names
        )
        w(f"| {k[0]} | {k[1]} | {k[2]} | {rec} | {mae} | {m5} | {ll} |")

    w("\n#### Totals: runs recovered on a return, failed fits on steady and drift\n")
    w("| | " + " | ".join(names) + " |")
    w("|---|" + "---|" * len(names))
    for cov in ("full", "diag"):
        ab = [r for r in rs if r["kind"] == "absent" and r["cov"] == cov]
        w(
            f"| {cov} recovered | "
            + " | ".join(
                f"{sum(not base.failed(r['m']) for r in ab if r['variant'] == n)}/"
                f"{sum(r['variant'] == n for r in ab)}"
                for n in names
            )
            + " |"
        )
        for kind in ("steady", "drift"):
            st = [r for r in rs if r["kind"] == kind and r["cov"] == cov]
            w(
                f"| {cov} {kind} failed | "
                + " | ".join(
                    str(sum(base.failed(r["m"]) for r in st if r["variant"] == n)) for n in names
                )
                + " |"
            )

    w(
        "\n#### Cells where a variant loses to A (MAE 2% and 5e-4 above, loglik 0.01 below, "
        "or more failed fits)\n"
    )
    cells = group(rs, lambda r: (r["cov"], r["d"], r["hl"], r["kind"], r.get("gap")))
    for n in names[1:]:
        lost = [k for k, v in cells.items() if n in v and loss(v["A"], v[n])]
        w(f"- **{n}**: {len(lost)} of {len(cells)}")
        for k in sorted(lost, key=str):
            v = cells[k]
            ma = base._mean([r["m"]["mae"] for r in v["A"]])
            mb = base._mean([r["m"]["mae"] for r in v[n]])
            fa = sum(base.failed(r["m"]) for r in v["A"])
            fb = sum(base.failed(r["m"]) for r in v[n])
            la = base._mean([r["m"].get("llgap", np.nan) for r in v["A"]])
            lb = base._mean([r["m"].get("llgap", np.nan) for r in v[n]])
            w(
                f"  - {k[0]} d={k[1]} hl={k[2]} {k[3]}{'' if k[4] is None else ' ' + str(k[4])}: "
                f"MAE {ma:.4f} -> {mb:.4f}, loglik gap {la:.3f} -> {lb:.3f}, failed {fa} -> {fb}"
            )
    if "E" in names:
        w(
            "\n#### F against E (the lazy prior's cost at 13 and 20 features): "
            "mean |ΔMAE|, largest\n"
        )
        diffs = []
        for k, v in cells.items():
            if "E" in v and "F" in v:
                for a, b in zip(v["F"], v["E"], strict=True):
                    if "mae" in a["m"] and "mae" in b["m"]:
                        diffs.append((abs(a["m"]["mae"] - b["m"]["mae"]), k))
        if diffs:
            diffs.sort(key=lambda t: t[0])
            w(
                f"- mean {np.mean([d for d, _ in diffs]):.2e}, "
                f"largest {diffs[-1][0]:.2e} at {diffs[-1][1]}"
            )
    return "\n".join(lines)


def warm_report(out: Path) -> str:
    rs = json.loads((out / "results-warm.json").read_text())
    lines = [
        "| cov | d | warm rows | failed fits A (of 6) | failed fits F (of 6) | MAE A | MAE F |"
    ]
    lines.append("|---|---|---|---|---|---|---|")
    g: dict = {}
    for r in rs:
        g.setdefault((r["cov"], r["d"], r["warm"]), {}).setdefault(r["variant"], []).append(r)
    for k in sorted(g):
        v = g[k]
        fa = sum(base.failed(r["m"]) for r in v["A"])
        ff = sum(base.failed(r["m"]) for r in v["F"])
        ma = base._mean([r["m"].get("mae", np.nan) for r in v["A"]])
        mf = base._mean([r["m"].get("mae", np.nan) for r in v["F"]])
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {fa} | {ff} | {ma:.4f} | {mf:.4f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--extra", action="append", default=[], help="NAME=SPEC, e.g. Fm=M:{d}")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.report:
        print(report(a.out))
        if (a.out / "results-warm.json").exists():
            print("\n#### The warm-up\n")
            print(warm_report(a.out))
        return
    exe = base.build()
    extra = dict(e.split("=", 1) for e in a.extra)
    js = warm_jobs() if a.warm else jobs(a.quick, extra)
    for j in js:
        j["exe"], j["work"] = str(exe), str(a.out)
    results = []
    fn = warm_cell if a.warm else base.cell
    with ProcessPoolExecutor(a.workers) as pool:
        for i, r in enumerate(pool.map(fn, js, chunksize=1)):
            results.extend(r)
            if i % 50 == 0:
                print(f"{i}/{len(js)}", file=sys.stderr, flush=True)
    name = "results-warm.json" if a.warm else "results-bd.json"
    if extra and not a.warm and (a.out / name).exists():
        # Extra variants alone: added to the results already there.
        old = json.loads((a.out / name).read_text())
        keep = [r for r in old if r["variant"] not in extra]
        results = keep + [r for r in results if r["variant"] in extra]
    (a.out / name).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
