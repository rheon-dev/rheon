"""The ``invert`` node (``crates/rheon-polars/src/formula/scalar.rs``)
against scipy's ``brentq``, on a seeded monotone closed form.

Two ways in. The example ``crates/rheon-polars/examples/invert_rows``
reads the node in its list form, with the columns, from stdin, and runs it
as a Polars expression over the frame; and ``rp.invert``, the Python name,
writes the same node from a Python function of the unknown and runs it in
``rp.stream.with_windows``. Either way the same function is written in
NumPy and solved by ``scipy.optimize.brentq`` at the same ``xtol`` and
``rtol``: each solver's answer is within its tolerance of a sign change of
its own computed function, so where the node certifies a root the two agree
to ``2 eps + xtol + rtol |x|``, ``eps`` the node's tolerance at the bracket.
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
import polars as pl
import pytest
from scipy.optimize import brentq

from rheon import polars as rp
from rheon.polars import _formula

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


def _rows(n: int = 1000, seed: int = 5632):
    """``n`` seeded rows: ``a`` in ``[0, 3]``, ``b`` in ``[0, 1]``, the
    target ``f`` at a root drawn in ``[-4.9, 4.9]``, and every twentieth
    target pushed past either end of the function's range or left null,
    with the reason each such row is withheld for."""
    rng = np.random.default_rng(seed)
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
    return a, b, roots, t, expect


def _held_to_brentq(out: dict, a, b, roots, t, expect) -> float:
    """Every row in range certified and within the bound of ``brentq``'s
    root, inside its bracket and near the drawn root; every row out of
    range withheld with the side named, where ``brentq`` refuses for want
    of a sign change. The largest gap from ``brentq``."""
    eps = XTOL  # the bracket holds 0
    worst = 0.0
    for i in range(len(a)):
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
    return worst


def test_the_node_agrees_with_brentq_wherever_it_certifies(invert_solver):
    """The node in its list form, run by the example over 1,000 seeded rows
    (:func:`_rows`), held to ``brentq`` (:func:`_held_to_brentq`)."""
    a, b, roots, t, expect = _rows()
    payload = {"node": NODE, "columns": {"a": a.tolist(), "b": b.tolist(), "t": t}}
    res = subprocess.run(
        [str(invert_solver)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    worst = _held_to_brentq(json.loads(res.stdout), a, b, roots, t, expect)
    assert worst < 6e-12, worst


def _python_f(x: pl.Expr) -> pl.Expr:
    """``x^3 + a x + exp(b x)`` in Polars, for ``rp.invert``."""
    return x**3 + pl.col("a") * x + (pl.col("b") * x).exp()


def test_rp_invert_agrees_with_brentq_wherever_it_certifies():
    """``rp.invert`` over the same 1,000 seeded rows, through
    ``rp.stream.with_windows`` beside a window operator, under ``detail``:
    it writes the example's node from a Python function of the unknown, and
    its roots are held to ``brentq`` as the example's are, within
    ``2 eps + xtol + rtol |x| + 5 eps_mach |x|`` on every certified row
    (about ``6e-12`` here, the roots in ``[-4.9, 4.9]``). The largest gap
    measured is far inside it."""
    expr = rp.invert(_python_f, "t", LO, HI, increasing=True, xtol=XTOL, rtol=RTOL, detail=True)
    (column,) = expr.meta.root_names()
    assert json.loads(column[len(_formula.PREFIX) :]) == NODE
    a, b, roots, t, expect = _rows()
    frame = pl.DataFrame({"a": a, "b": b, "t": pl.Series(t, dtype=pl.Float64)})
    got = rp.stream.with_windows(frame, d=expr, level=rp.ewm_mean("a", half_life=4.0))
    assert got.height == len(a)
    detail = got["d"].struct.unnest()
    out = {name: detail[name].to_list() for name in detail.columns}
    worst = _held_to_brentq(out, a, b, roots, t, expect)
    assert worst < 6e-12, worst
    # The plain form gives the detail's roots, to the bit.
    plain = rp.stream.with_windows(
        frame,
        x=rp.invert(_python_f, "t", LO, HI, increasing=True, xtol=XTOL, rtol=RTOL),
        level=rp.ewm_mean("a", half_life=4.0),
    )
    assert plain["x"].to_list() == out["root"]
