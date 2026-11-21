"""Mitigation 1: variance targeting. ``omega`` is no longer estimated by the
likelihood: it is set from the mean of ``r^2`` and the persistence, and only
``alpha`` and ``beta`` take Gauss-Newton steps (garch.py, ``target``).
"running" uses the running mean over every row so far; "warmup" fixes it at
the mean over the 75 buffered rows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_garch  # noqa: E402

VARIANTS: dict[str, dict[str, object]] = {
    "target_running": {"target": "running"},
    "target_warmup": {"target": "warmup"},
}

if __name__ == "__main__":
    grid_garch.main(VARIANTS)
