"""Mitigation 2: a cap on each Gauss-Newton step, per free parameter, at a
fraction of that parameter's magnitude, with the projection kept
(garch.py, ``step_cap``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_garch  # noqa: E402

VARIANTS: dict[str, dict[str, object]] = {
    "cap_0.5": {"step_cap": 0.5},
    "cap_0.1": {"step_cap": 0.1},
}

if __name__ == "__main__":
    grid_garch.main(VARIANTS)
