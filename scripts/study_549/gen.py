"""PLAN 549 study (not a test): the series the GARCH harness reads.

Writes `.cache/study549/data/<name>.f64` (raw little-endian f64) and
`.cache/study549/manifest.csv` (name, vol, dist, start_rows, set). The
realistic "hi" truths are the oracle grid's own (tests/test_second_opinion_
garch.py, the same seeds, so its counts can be confirmed); "lo" truths have
persistence 0.90. The degenerate sets are i.i.d. returns (no GARCH effect)
and the realistic truths started on 50 rows.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
import test_second_opinion_garch as oracle  # noqa: E402

OUT = ROOT / ".cache" / "study549"
N = 10_000
SEEDS = range(50)

LO = {
    "garch": ("garch", "normal", (0.10, 0.10, 0.80)),
    "gjr": ("gjr", "normal", (0.10, 0.05, 0.10, 0.80)),
    "egarch": ("egarch", "normal", (0.0, 0.20, -0.10, 0.90)),
    "garch_t": ("garch", "t", (0.10, 0.10, 0.80)),
    "garch_skewt": ("garch", "skewt", (0.10, 0.10, 0.80)),
    "gjr_t": ("gjr", "t", (0.10, 0.05, 0.10, 0.80)),
}


def simulate_lo(form: str, n: int, seed: int) -> np.ndarray:
    vol, dist, th = LO[form]
    rng = np.random.default_rng([549, sorted(LO).index(form), seed])
    burn = 1000
    z = oracle.innovations(dist, n + burn, rng)
    out = np.empty(n + burn)
    if vol == "egarch":
        om, a, g, b = th
        lv, zp = om / (1.0 - b), 0.0
        for t in range(n + burn):
            lv = om + a * (abs(zp) - oracle.ABS_Z) + g * zp + b * lv
            out[t] = math.exp(0.5 * lv) * z[t]
            zp = z[t]
    else:
        gjr = vol == "gjr"
        om, a, g, b = th if gjr else (th[0], th[1], 0.0, th[2])
        s = om / (1.0 - a - 0.5 * g - b)
        e = 0.0
        for t in range(n + burn):
            s = om + (a + (g if e < 0 else 0.0)) * e * e + b * s
            e = math.sqrt(s) * z[t]
            out[t] = e
    return out[burn:]


def main() -> None:
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    rows = []

    def put(name: str, r: np.ndarray, vol: str, dist: str, starts, kind: str) -> None:
        p = data / f"{name}.f64"
        if not p.exists():
            np.asarray(r, dtype="<f8").tofile(p)
        for s in starts:
            rows.append(f"{name},{vol},{dist},{s},{kind}")

    for form, (vol, dist, _, _) in oracle.FORMS.items():
        starts = (2000,) if vol == "figarch" else (50, 500, 1000, 2000)
        for seed in SEEDS:
            r = oracle.simulate(form, N, seed)
            put(f"hi_{form}_{seed}", r, vol, dist, starts, "hi")
    for form, (vol, dist, _) in LO.items():
        for seed in SEEDS:
            put(
                f"lo_{form}_{seed}",
                simulate_lo(form, N, seed),
                vol,
                dist,
                (50, 500, 1000, 2000),
                "lo",
            )
    # no GARCH effect: i.i.d. returns, each form on its own innovations
    for form, (vol, dist, _) in LO.items():
        for seed in SEEDS:
            rng = np.random.default_rng([5490, sorted(LO).index(form), seed])
            put(
                f"iid_{form}_{seed}",
                oracle.innovations(dist, N, rng),
                vol,
                dist,
                (50, 500, 2000),
                "iid",
            )
    (OUT / "manifest.csv").write_text(
        "name,vol,dist,start,set\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    print(len(rows), "runs")


if __name__ == "__main__":
    main()
