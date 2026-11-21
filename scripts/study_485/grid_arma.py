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
    # request A10, part 3: Student-t(5) innovations, unit SD, two AR roots
    "t5_phi080": [(10000, math.inf)],
    "t5_phi095": [(10000, math.inf)],
    # their Gaussian controls: the same truths, normal innovations
    "n_phi080": [(10000, math.inf)],
    "n_phi095": [(10000, math.inf)],
    # request A11, remedy (c): the robust score (arma.py, robust_c) on phi 0.8
    "t5_phi080_huber2": [(10000, math.inf)],
    "n_phi080_huber2": [(10000, math.inf)],
    # request A11, part 3: the bounded batch start with the robust score (a + c)
    "t5_phi080_batch500_huber2": [(10000, math.inf)],
    "n_phi080_batch500_huber2": [(10000, math.inf)],
    # request A12, part 1: the same with the core's batch optimizer
    "t5_phi080_batch500core_huber2": [(10000, math.inf)],
    "n_phi080_batch500core_huber2": [(10000, math.inf)],
}
# the truth and the noise per grid; part 4's grids keep arma.TRUTH, Gaussian
SETTINGS: dict[str, dict[str, object]] = {
    "t5_phi080": {"truth": (0.8, 0.3, 1.0), "nu": 5.0},
    "t5_phi095": {"truth": (0.95, 0.3, 1.0), "nu": 5.0},
    "n_phi080": {"truth": (0.8, 0.3, 1.0)},
    "n_phi095": {"truth": (0.95, 0.3, 1.0)},
    "t5_phi080_huber2": {"truth": (0.8, 0.3, 1.0), "nu": 5.0, "robust_c": 2.0},
    "n_phi080_huber2": {"truth": (0.8, 0.3, 1.0), "robust_c": 2.0},
    "t5_phi080_batch500_huber2": {
        "truth": (0.8, 0.3, 1.0),
        "nu": 5.0,
        "robust_c": 2.0,
        "batch_start": 500,
    },
    "n_phi080_batch500_huber2": {"truth": (0.8, 0.3, 1.0), "robust_c": 2.0, "batch_start": 500},
    "t5_phi080_batch500core_huber2": {
        "truth": (0.8, 0.3, 1.0),
        "nu": 5.0,
        "robust_c": 2.0,
        "batch_start": 500,
        "batch_opt": "core",
    },
    "n_phi080_batch500core_huber2": {
        "truth": (0.8, 0.3, 1.0),
        "robust_c": 2.0,
        "batch_start": 500,
        "batch_opt": "core",
    },
}


def key(n: int, h: float) -> tuple[int, float]:
    return (n, h if math.isfinite(h) else -1.0)


def load(out: Path) -> list[dict]:
    if not out.exists():
        return []
    return [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(out: Path, cases: list[tuple[int, float]], settings: dict[str, object]) -> None:
    finished = {(int(r["seed"]), int(r["n"]), r["half_life"]) for r in load(out)}
    with out.open("a", encoding="utf-8") as f:
        for n, h in cases:
            for seed in SEEDS:
                if (seed, *key(n, h)) in finished:
                    continue
                rec = arma.compare(n, seed, half_life=h, **settings)  # type: ignore[arg-type]
                f.write(json.dumps(rec) + "\n")
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
        # the per-seed bounds proposed in the record's section 15.2
        se_ok = sum(all(abs(r[f"{k}_diff_in_se"]) <= 1.0 for k in arma.NAMES) for r in rows)
        rms_ok = int((gap <= 0.025).sum())
        both = sum(
            all(abs(r[f"{k}_diff_in_se"]) <= 1.0 for k in arma.NAMES)
            and r["pred_rms_gap_second_half"] <= 0.025
            for r in rows
        )
        worst = max(rows, key=lambda r: max(abs(r[f"{k}_diff_in_se"]) for k in arma.NAMES))
        wk = max(arma.NAMES, key=lambda k: abs(worst[f"{k}_diff_in_se"]))
        print(
            f"<!-- n {n} h {hl}: section 15.2 bounds: all within 1 SE {se_ok}; rms gap "
            f"<= 0.025 {rms_ok}; both {both}; worst seed {int(worst['seed'])} {wk} "
            f"{worst[f'{wk}_diff_in_se']:+.3g} SE -->"
        )


if __name__ == "__main__":
    outdir, which = Path(sys.argv[1]), sys.argv[2]
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{which}.jsonl"
    if "--summary" in sys.argv[3:]:
        summary(out, CASES[which])
    else:
        run(out, CASES[which], SETTINGS.get(which, {}))
