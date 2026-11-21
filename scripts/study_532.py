"""Study 532: resumable time-series features held in a state, against Polars.

Run with a Python that has numpy and the pinned Polars (2.0.0):
``python scripts/study_532.py``. It prints every table of
docs/records/STUDY-532-FEATURES.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studies.study_532.report import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
