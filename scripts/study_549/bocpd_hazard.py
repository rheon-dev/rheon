"""PLAN 549 study (not a test), part (b): what clock units would do to
`bocpd`'s outputs. A number `hazard` counts rows (H = 1/hazard after each
row); a duration is the clock form (h = 1 - exp(-d/tau) before each row).
Each stream is run both ways on the same clock, with breaks generated either
per row (geometric in rows, the paper's prior) or per unit of time (a
Poisson process on the clock), on a regular clock (one second a row), an
irregular one (exponential gaps, mean one second) and one with overnight
gaps (one-second rows in sessions of 2,000, an hour between sessions).
Reported: the mean predictive log-likelihood a row (higher is better), the
mean |difference| in p_change, and how often run_mode differs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_bocpd import _START, run_clock  # noqa: E402

N = 4000
LAM = 200.0  # rows, or seconds


def clocks(rng: np.random.Generator) -> dict[str, np.ndarray]:
    regular = np.arange(N, dtype=float)
    irregular = np.cumsum(rng.exponential(1.0, N))
    sessions = np.arange(N, dtype=float) + 3600.0 * (np.arange(N) // 2000)
    return {"regular": regular, "irregular": irregular, "sessions": sessions}


def levels(secs: np.ndarray, per: str, rng: np.random.Generator) -> np.ndarray:
    """A level that jumps at breaks, N(0, 3^2) each, plus N(0,1) noise."""
    if per == "row":
        brk = rng.uniform(size=N) < 1.0 / LAM
    else:
        gaps = np.diff(secs, prepend=secs[0])
        brk = rng.uniform(size=N) < -np.expm1(-gaps / LAM)
    seg = np.cumsum(brk)
    mu = rng.normal(0.0, 3.0, seg.max() + 1)
    return mu[seg] + rng.normal(0.0, 1.0, N)


def main() -> None:
    print("clock      breaks  hazard  loglik/row  |dp_change|  run_mode differs")
    for clock in ("regular", "irregular", "sessions"):
        for per in ("row", "time"):
            ll = {"rows": [], "clock": []}
            dp, dm = [], []
            for seed in range(5):
                rng = np.random.default_rng([549, seed])
                secs = clocks(rng)[clock]
                x = levels(secs, per, rng)
                a = run_clock(x, _START + secs, LAM)
                b = run_clock(x, _START + secs, f"{int(LAM)}s")
                ll["rows"].append(float(a["loglik"].mean()))
                ll["clock"].append(float(b["loglik"].mean()))
                dp.append(float((a["p_change"] - b["p_change"]).abs().mean()))
                dm.append(float((a["run_mode"] != b["run_mode"]).mean()))
            for h in ("rows", "clock"):
                cells = f"{np.mean(ll[h]):10.5f}  {np.mean(dp):11.2e}  {np.mean(dm):.3f}"
                print(f"{clock:10s} {per:6s}  {h:6s}  {cells}")


if __name__ == "__main__":
    main()
