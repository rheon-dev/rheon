#!/usr/bin/env python
"""Study 511b (C11): PLAN tasks 517-525 as numpy online forms against their corrected oracles.
Prints every table of docs/records/STUDY-511-FEATURES-2.md.

    python scripts/study_511b.py [--only 517 518 ...] [--seeds N]

Runs in a venv of its own (numpy, scipy, statsmodels, scikit-learn, polars, hmmlearn,
filterpy, dieboldmariano, torch + bayesian-changepoint), not `uv sync`: it builds no Rust
and is not collected by pytest.
"""

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

MODULES = {
    "517": "studies.study_511b.t517",
    "518": "studies.study_511b.t518",
    "519": "studies.study_511b.t519",
    "520": "studies.study_511b.t520",
    "521": "studies.study_511b.t521",
    "522": "studies.study_511b.t522",
    "523": "studies.study_511b.t523",
    "524": "studies.study_511b.t524",
    "525": "studies.study_511b.t525",
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
