"""Study 492, B12 part 1: is GARCH(2,2)'s failure against arch the weak identification of beta1 and beta2?
(docs/records/STUDY-492-GARCH-FAMILY.md, part 6). Python and numpy only; arch is the oracle.

    study_492_b12.py run TRUTH OUTDIR SEED...   # one truth (a, b or c), seeds, appended to OUTDIR/TRUTH.jsonl
    study_492_b12.py summary OUTDIR

Reuses study_492_b11's comparison unchanged (`compare_g22`: part 1's targeted step and part 5's PACF step against
arch's Gaussian GARCH(2,2) fit, same rows) and only swaps the simulating truth. Parameters are
(omega, alpha1, alpha2, beta1, beta2), omega set so the unconditional variance is 1:
    a: (0.20, 0.05, 0.00, 0.45, 0.30)   a strong second lag: beta2 / beta1 = 0.67; persistence 0.80
    b: (0.15, 0.05, 0.05, 0.45, 0.30)   alpha2 non-zero, same betas; persistence 0.85
    c: (0.03, 0.05, 0.05, 0.45, 0.42)   the reference truth of the earlier GARCH(2,2) runs; persistence 0.97
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_492_b11 as B  # noqa: E402

TRUTHS = {
    "a": np.array([0.20, 0.05, 0.00, 0.45, 0.30]),
    "b": np.array([0.15, 0.05, 0.05, 0.45, 0.30]),
    "c": np.array([0.03, 0.05, 0.05, 0.45, 0.42]),
}
N = 10_000


def run(truth: str, outdir: str, seeds: list[int]) -> None:
    B.G22_TRUTH = TRUTHS[truth]  # compare_g22 reads the module's truth
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{truth}.jsonl").open("a") as fh:
        for s in seeds:
            t = time.perf_counter()
            d = B.compare_g22(s, N)
            d["truth"], d["seconds"] = TRUTHS[truth].tolist(), time.perf_counter() - t
            fh.write(json.dumps(d) + "\n")
            fh.flush()
            print(truth, s, f"{d['seconds']:.1f}s", flush=True)


def summary(outdir: str) -> None:
    for truth in TRUTHS:
        p = Path(outdir) / f"{truth}.jsonl"
        if not p.exists():
            continue
        rows = sorted((json.loads(x) for x in p.read_text().splitlines()), key=lambda d: d["seed"])
        se = np.array([d["se"] for d in rows])
        print(
            f"truth {truth} {TRUTHS[truth].tolist()}: {len(rows)} seeds, n {N}; arch converged "
            f"{sum(d['arch_converged'] for d in rows)}; arch corr(b1, b2) median "
            f"{np.median([d['arch_corr_b1b2'] for d in rows]):.3f} (weakest {max(d['arch_corr_b1b2'] for d in rows):.3f}); "
            f"arch SE b1 / b2 median {np.median(se[:, 3]):.3f} / {np.median(se[:, 4]):.3f}"
        )
        for tag in ("target", "pacf"):
            v = [d[tag] for d in rows]
            G = np.abs(np.array([x["gap_se"] for x in v]))
            pg = [x["path_max_rel_last1000"] for x in v]
            print(
                f"  {tag}: within 2 SE all {sum(x['within2_all'] for x in v)}, without omega "
                f"{sum(x['within2_noomega'] for x in v)}; persistence {sum(abs(x['pers_gap_se']) <= 2 for x in v)}, "
                f"beta sum {sum(abs(x['bsum_gap_se']) <= 2 for x in v)}; b1 only {int((G[:, 3] <= 2).sum())}, "
                f"b2 only {int((G[:, 4] <= 2).sum())}; median |gap| (SE) "
                + ", ".join(f"{np.median(G[:, j]):.2f}" for j in range(5))
                + f"\n    path gap max over last 1000: median {100 * np.median(pg):.1f}% worst {100 * np.max(pg):.1f}% "
                f"(seed {rows[int(np.argmax(pg))]['seed']}); ll gap median {np.median([x['ll_gap'] for x in v]):.2f} "
                f"worst {np.min([x['ll_gap'] for x in v]):.1f}"
            )


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "summary":
        summary(a[1])
    else:
        run(a[1], a[2], [int(s) for s in a[3:]])
