"""The ARMA(1,1) grids: the online estimator against statsmodels' batch fit over
fifty seeds (docs/records/ORACLE-STUDY-485.md, part 4).

    python grid_arma.py OUTDIR grid|decay      # run, resuming after the last case written
    python grid_arma.py OUTDIR grid|decay --summary

``grid``: no decay, 2,000 and 10,000 rows. ``decay``: 10,000 rows at half-lives
500, 5,000 and none, each against the weighted fit of the same half-life and
against the unweighted one. One JSON line per case, flushed as it finishes.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arma  # noqa: E402

SEEDS = range(485000, 485050)
CASES = {
    "grid": [(n, math.inf) for n in (2000, 10000)],
    "decay": [(10000, h) for h in (500.0, 5000.0, math.inf)],
}


def key(n: int, h: float) -> tuple[int, float]:
    return (n, h if math.isfinite(h) else -1.0)


def load(out: Path) -> list[dict]:
    if not out.exists():
        return []
    return [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(out: Path, cases: list[tuple[int, float]]) -> None:
    finished = {(int(r["seed"]), int(r["n"]), r["half_life"]) for r in load(out)}
    with out.open("a", encoding="utf-8") as f:
        for n, h in cases:
            for seed in SEEDS:
                if (seed, *key(n, h)) in finished:
                    continue
                f.write(json.dumps(arma.compare(n, seed, half_life=h)) + "\n")
                f.flush()
                print(f"{out.stem}: n {n} half-life {h} seed {seed} done", flush=True)


def counts(rows: list[dict], suffix: str) -> str:
    cells = []
    for name in arma.NAMES:
        a = np.abs([r[f"{name}{suffix}"] for r in rows])
        cells.append(f"{np.median(a):.3g} / {a.max():.3g} / {int((a > 3).sum())}")
    within2 = sum(all(abs(r[f"{k}{suffix}"]) <= 2 for k in arma.NAMES) for r in rows)
    return " | ".join(cells) + f" | {within2}"


def summary(out: Path, cases: list[tuple[int, float]]) -> None:
    recs = load(out)
    print(
        "| rows | half-life | against | phi: median / max / over 3 | theta | sigma2 "
        "| all within 2 | statsmodels converged |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for n, h in cases:
        kn, kh = key(n, h)
        rows = [r for r in recs if int(r["n"]) == kn and r["half_life"] == kh]
        conv = sum(r["sm_converged"] == 1 for r in rows)
        hl = "none" if kh < 0 else f"{kh:,.0f}"
        print(f"| {n:,} | {hl} | unweighted fit, its SE | {counts(rows, '_diff_in_se')} | {conv} |")
        if out.stem == "decay":
            vs_w = counts(rows, "_vs_weighted_in_se_w")
            print(f"| {n:,} | {hl} | weighted fit, scaled SE | {vs_w} | |")
            print(
                f"| {n:,} | {hl} | weighted fit against unweighted | "
                f"{counts(rows, '_weighted_vs_batch_in_se')} | |"
            )
        gap = np.array([r["pred_rms_gap_second_half"] for r in rows])
        ratio = np.array([r["online_mse_second_half"] / r["batch_mse_second_half"] for r in rows])
        print(
            f"<!-- n {n} h {hl}: seeds {len(rows)}; prediction rms gap median "
            f"{np.median(gap):.3g} max {gap.max():.3g}; mse ratio median "
            f"{np.median(ratio):.4f} max {ratio.max():.4f} -->"
        )


if __name__ == "__main__":
    outdir, which = Path(sys.argv[1]), sys.argv[2]
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{which}.jsonl"
    if "--summary" in sys.argv[3:]:
        summary(out, CASES[which])
    else:
        run(out, CASES[which])
