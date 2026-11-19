"""The package a benchmark runs: this build's `rheon`, or a release before 2.0.

A benchmark runs under the last release and under this build in turn
(`docs/contributing/cut-a-release.md`, step 2), so it imports whichever of the
two the interpreter has. `rheon` is first; `polars_online` is the name of every
release before 2.0, and is tried only where `rheon` is not installed. The two
differ in four ways this module hides:

- the module: `rheon.polars` against `polars_online`;
- the frame namespace: `lf.rheon` against `lf.online`;
- the thread-count variable: `RHEON_MAX_THREADS` against
  `POLARS_ONLINE_MAX_THREADS` (this build reads the old name once, with a
  warning, so a benchmark sets the one its build reads);
- the distribution's name in the line a benchmark prints first.

    from rheon_import import rp, namespace

`PRELUDE` is the same choice as source text, for the benchmarks that run their
workload in a fresh interpreter: it binds `po` (the module) and `ns` (the
namespace's name) before the workload's own lines.

Standard library only, besides the package it imports.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Any

#: Whether this interpreter has this build (2.0 and later) rather than a release before it.
IS_RHEON: bool = importlib.util.find_spec("rheon") is not None

#: The distribution, as `pip` names it.
PACKAGE: str = "rheon" if IS_RHEON else "polars-online"

#: The frame namespace: `lf.<NAMESPACE>.fit_predict(...)`.
NAMESPACE: str = "rheon" if IS_RHEON else "online"

#: The environment variable that sets the bank's thread count.
THREADS_VAR: str = "RHEON_MAX_THREADS" if IS_RHEON else "POLARS_ONLINE_MAX_THREADS"

#: `rheon.polars` or `polars_online`: the module with `ModelBank`, `spec` and the window ops.
rp: ModuleType = importlib.import_module("rheon.polars" if IS_RHEON else "polars_online")

#: The same choice as source text, for a workload run in a fresh interpreter.
PRELUDE = (
    "import importlib, importlib.util\n"
    "_new = importlib.util.find_spec('rheon') is not None\n"
    "po = importlib.import_module('rheon.polars' if _new else 'polars_online')\n"
    "ns = 'rheon' if _new else 'online'\n"
)


def namespace(frame: Any) -> Any:
    """`frame.rheon` for this build, `frame.online` for a release before 2.0."""
    return getattr(frame, NAMESPACE)
