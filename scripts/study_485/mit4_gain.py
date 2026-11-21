"""Mitigation 4: a gain that shrinks more slowly than ``1/(t + 75)`` (part 2's
plain estimator): ``1/sqrt(t + 75)``, and ``1/(t + 75)`` floored at 1e-3. The
information matrix keeps its ``1/(t + 75)`` average (garch.py, ``gain``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_garch  # noqa: E402

VARIANTS: dict[str, dict[str, object]] = {
    "gain_sqrt": {"gain": "sqrt"},
    "gain_floor_1e-3": {"gain": "floor", "gain_floor": 1e-3},
}

if __name__ == "__main__":
    grid_garch.main(VARIANTS)
