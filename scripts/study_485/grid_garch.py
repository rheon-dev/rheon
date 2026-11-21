"""The GARCH(1,1) grid: the online estimator against arch's batch fit over
fifty seeds and two row counts (docs/records/ORACLE-STUDY-485.md, part 2).

    python grid_garch.py OUT.jsonl            # run, resuming after the last finished case
    python grid_garch.py OUT.jsonl --summary  # the distribution, as the record's table

One JSON line per (seed, rows), appended and flushed as each case finishes,
so a stopped run resumes from the first case not yet in the file.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import garch  # noqa: E402

TRUTH = (0.02, 0.08, 0.90)
SEEDS = range(485000, 485050)
SIZES = (2000, 10000)
FIELDS = (
    "omega_diff_in_se",
    "alpha_diff_in_se",
    "beta_diff_in_se",
    "llf_diff_per_row",
    "pred_median_rel_second_half",
)


def done(out: Path) -> set[tuple[int, int]]:
    if not out.exists():
        return set()
    keys = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            keys.add((int(rec["seed"]), int(rec["n"])))
    return keys


def run(out: Path) -> None:
    finished = done(out)
    with out.open("a", encoding="utf-8") as f:
        for n in SIZES:
            for seed in SEEDS:
                if (seed, n) in finished:
                    continue
                rec = garch.compare(n, seed, TRUTH)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"seed {seed} n {n} done", flush=True)


def summary(out: Path) -> None:
    recs = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    for n in SIZES:
        rows = [r for r in recs if int(r["n"]) == n]
        converged = sum(r["arch_converged"] == 0 for r in rows)
        print(f"rows {n}: {len(rows)} seeds; arch converged on {converged}")
        print(f"  {'quantity':30s} {'median':>10s} {'max':>10s} {'>3 SE':>6s}")
        for k in FIELDS:
            v = np.array([r[k] for r in rows])
            a = np.abs(v)
            over = str(int(np.sum(a > 3.0))) if k.endswith("_se") else "-"
            print(f"  {k:30s} {np.median(a):10.4g} {np.max(a):10.4g} {over:>6s}")
        either = sum(any(abs(r[k]) > 3.0 for k in FIELDS[:3]) for r in rows)
        within2 = sum(all(abs(r[k]) <= 2.0 for k in FIELDS[:3]) for r in rows)
        neg = sum(r["llf_diff_per_row"] < 0 for r in rows)
        print(f"  seeds with any parameter over 3 SE: {either}; all within 2 SE: {within2}")
        print(f"  seeds where the online parameters out-score arch's fit: {neg}")


if __name__ == "__main__":
    path = Path(sys.argv[1])
    if "--summary" in sys.argv[2:]:
        summary(path)
    else:
        run(path)
