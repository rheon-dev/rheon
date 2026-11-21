#!/usr/bin/env python
"""Study 511 (C10): five additive features of PLAN tasks 511-525 as numpy
online forms, each against its corrected oracle. Prints every table of
docs/records/STUDY-511-FEATURES-1.md.

    python scripts/study_511.py [--only 511 512 513 515 516] [--seeds N]

Runs in a venv of its own (numpy, scipy, statsmodels, scikit-learn, crepes,
mapie), not `uv sync`: it builds no Rust and is not collected by pytest.
"""

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

MODULES = {
    "511": "studies.study_511.shrink",
    "512": "studies.study_511.hc",
    "513": "studies.study_511.scores",
    "515": "studies.study_511.conformal",
    "516": "studies.study_511.constrain",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(MODULES))
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    for t in a.only:
        print(f"\n## Task {t}")
        importlib.import_module(MODULES[t]).report(seeds=a.seeds)


if __name__ == "__main__":
    main()
