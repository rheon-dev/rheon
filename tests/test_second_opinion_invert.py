"""The ``invert`` node (``crates/rheon-polars/src/formula/scalar.rs``)
against scipy's ``brentq``, on a seeded monotone closed form.

The node has no Python name yet, so the example
``crates/rheon-polars/examples/invert_rows`` reads it in its list form,
with the columns, from stdin, and runs it as a Polars expression over the
frame. Here the same function is written in NumPy and solved by
``scipy.optimize.brentq`` at the same ``xtol`` and ``rtol``: each solver's
answer is within its tolerance of a sign change of its own computed
function, so where the node certifies a root the two agree to
``2 eps + xtol + rtol |x|``, ``eps`` the node's tolerance at the bracket.
"""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

TIER = "essential"

REPO = Path(__file__).resolve().parent.parent

XTOL = 2e-12
RTOL = 4 * float(np.finfo(float).eps)
EPS_MACH = float(np.finfo(float).eps)
LO, HI = -5.0, 5.0


@contextlib.contextmanager
def _build_lock(path: Path) -> Iterator[None]:
    """An exclusive lock across pytest's worker processes, where the
    platform has ``flock``: one worker's build must not replace the file
    another is copying."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@pytest.fixture(scope="session")
def invert_solver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The example ``invert_rows``, built once a session, its path read
    from cargo's own artifact message and the file copied for this process."""
    cmd = [
        "cargo",
        "build",
        "-q",
        "-p",
        "rheon-polars",
        "--example",
        "invert_rows",
        "--message-format=json-render-diagnostics",
    ]
    with _build_lock(REPO / "target" / ".invert-example-build.lock"):
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", cwd=str(REPO), check=False
            )
        except FileNotFoundError:
            pytest.fail("cargo is not on PATH; `source scripts/env.sh` first")
        assert res.returncode == 0, res.stderr
        exes = [
            msg["executable"]
            for line in res.stdout.splitlines()
            if line.startswith("{")
            and (msg := json.loads(line)).get("reason") == "compiler-artifact"
            and msg.get("executable")
            and msg["target"]["name"] == "invert_rows"
        ]
        assert len(exes) == 1, f"expected one example artifact, got {exes}:\n{res.stdout}"
        own = tmp_path_factory.mktemp("invert-example") / Path(exes[0]).name
        shutil.copy2(exes[0], own)
    return own


def f(x: float, a: float, b: float) -> float:
    """``x^3 + a x + exp(b x)``: increasing for ``a, b >= 0``."""
    return float(x**3 + a * x + np.exp(b * x))


#: The same function as the node's list form, the unknown ``x``.
X = ["unknown"]
NODE = [
    "invert",
    ["+", ["+", ["**", X, ["lit", 3]], ["*", ["col", "a"], X]], ["exp", ["*", ["col", "b"], X]]],
    ["col", "t"],
    ["lit", LO],
    ["lit", HI],
    {"increasing": True, "xtol": XTOL, "rtol": RTOL, "detail": True},
]


def test_the_node_agrees_with_brentq_wherever_it_certifies(invert_solver):
    """1,000 seeded rows: ``a`` in ``[0, 3]``, ``b`` in ``[0, 1]``, the
    target ``f`` at a root drawn in ``[-4.9, 4.9]``, and every twentieth
    target pushed past either end of the function's range or left null.
    Every row in range is certified and within the bound of ``brentq``'s
    root; the rows out of range are withheld with the side named, where
    ``brentq`` refuses for want of a sign change."""
    rng = np.random.default_rng(5632)
    n = 1000
    a = rng.uniform(0.0, 3.0, n)
    b = rng.uniform(0.0, 1.0, n)
    roots = rng.uniform(-4.9, 4.9, n)
    t: list[float | None] = [f(r, ai, bi) for r, ai, bi in zip(roots, a, b, strict=True)]
    expect: list[str | None] = [None] * n
    for i in range(0, n, 20):
        side = (i // 20) % 3
        if side == 0:
            t[i], expect[i] = f(LO, a[i], b[i]) - 1.0, "below_range"
        elif side == 1:
            t[i], expect[i] = f(HI, a[i], b[i]) + 1.0, "above_range"
        else:
            t[i], expect[i] = None, "input_missing"
    payload = {"node": NODE, "columns": {"a": a.tolist(), "b": b.tolist(), "t": t}}
    res = subprocess.run(
        [str(invert_solver)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    out = json.loads(res.stdout)
    eps = XTOL  # the bracket holds 0
    worst = 0.0
    for i in range(n):
        x, why = out["root"][i], out["reason"][i]
        assert out["evaluations"][i] <= out["bound"][i]
        assert why == expect[i], (i, why, expect[i])
        if expect[i] is not None:
            assert x is None
            if t[i] is not None:
                with pytest.raises(ValueError, match="different signs"):
                    brentq(lambda z, i=i: f(z, a[i], b[i]) - t[i], LO, HI)
            continue
        ref = brentq(lambda z, i=i: f(z, a[i], b[i]) - t[i], LO, HI, xtol=XTOL, rtol=RTOL)
        gap = abs(x - ref)
        assert gap <= 2 * eps + XTOL + RTOL * abs(ref) + 5 * EPS_MACH * abs(x), (i, x, ref)
        assert out["lo"][i] <= x <= out["hi"][i]
        assert math.isclose(x, roots[i], abs_tol=1e-9)
        worst = max(worst, gap)
    assert worst < 6e-12, worst
