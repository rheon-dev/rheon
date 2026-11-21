"""Study 532b: rolling statistics that keep no window of rows, against Polars.

Run with a Python that has numpy, scipy and the pinned Polars (2.0.0):
``python scripts/study_532b.py`` prints every table of
docs/records/STUDY-532-WINDOWLESS.md. ``--quick`` runs smaller frames;
``--part=N`` (repeatable) runs only part N (1 buckets, 2 exponential weights,
3 lags, 4 chunk and resume invariance and cost, 5 lag size checks).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studies.study_532b.report import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
