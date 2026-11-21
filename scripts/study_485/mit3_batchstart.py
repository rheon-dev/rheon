"""Mitigation 3: a longer batch start. The first 500 rows are buffered and
fitted by arch (standing in for a batch fit the model would own); the online
steps start from that fit, with the information matrix built on those 500
rows and ``t0 = 500`` (garch.py, ``batch_start``). The 500 rows emit nothing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_garch  # noqa: E402

VARIANTS: dict[str, dict[str, object]] = {
    "batch_500": {"batch_start": True, "warm": 500, "t0": 500.0},
}

if __name__ == "__main__":
    grid_garch.main(VARIANTS)
