"""The guarded root-finder (``crates/rheon-core/src/root.rs``) against scipy's
``brentq``, on Black-76's implied volatility and a family of cubics.

The solver has no Python binding yet, so the example
``crates/rheon-core/examples/root_black76`` runs it over rows written to its
stdin, one solve a line, with the function written in Rust as a user would
write it (``black76.rs``, beside it). Here the same function is written in
NumPy with ``scipy.special.ndtr`` and solved by ``scipy.optimize.brentq`` at
the same ``xtol`` and ``rtol``. Each solver's answer is within its tolerance
of a sign change of its own computed function, so where the solver certifies
its root (the price's rounding moves the root by less than the tolerance)
the two agree to ``2 eps + xtol + rtol |x|``, ``eps`` the solver's tolerance
at the bracket. The rows it withholds are those where the price carries no
more digits of the volatility than that.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import ndtr

TIER = "essential"

REPO = Path(__file__).resolve().parent.parent

#: The solver's defaults, and scipy's ``rtol`` default: four machine epsilons.
XTOL = 2e-12
RTOL = 4 * float(np.finfo(float).eps)
EPS_MACH = float(np.finfo(float).eps)

#: The volatility bracket.
LO, HI = 1e-4, 5.0


@contextlib.contextmanager
def _build_lock(path: Path) -> Iterator[None]:
    """An exclusive lock across pytest's worker processes, where the
    platform has ``flock`` (as ``conftest.py``'s for the command line): one
    worker's build must not replace the file another is copying."""
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
def root_solver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The example ``root_black76``, built once a session, its path read
    from cargo's own artifact message and the file copied for this process."""
    cmd = [
        "cargo",
        "build",
        "-q",
        "-p",
        "rheon-core",
        "--example",
        "root_black76",
        "--message-format=json-render-diagnostics",
    ]
    with _build_lock(REPO / "target" / ".root-example-build.lock"):
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
            and msg["target"]["name"] == "root_black76"
        ]
        assert len(exes) == 1, f"expected one example artifact, got {exes}:\n{res.stdout}"
        own = tmp_path_factory.mktemp("root-example") / Path(exes[0]).name
        shutil.copy2(exes[0], own)
    return own


def solve(exe: Path, lines: list[str]) -> list[tuple[float, float, float, int, int, str]]:
    """The solver's answer to each line: root (NaN where withheld), the
    bracket, its evaluations, their bound and ``ok`` or the reason."""
    res = subprocess.run(
        [str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True
    )
    out = []
    for row in res.stdout.splitlines():
        root, lo, hi, n, bound, reason = row.split()
        out.append((float(root), float(lo), float(hi), int(n), int(bound), reason))
    assert len(out) == len(lines)
    return out


def black76(sigma: float, fwd: float, strike: float, tau: float, df: float) -> float:
    """Black-76's call, in NumPy and scipy's ``ndtr``."""
    v = sigma * np.sqrt(tau)
    d1 = (np.log(fwd / strike) + 0.5 * v * v) / v
    return float(df * (fwd * ndtr(d1) - strike * ndtr(d1 - v)))


def black76_less(
    sigma: float, fwd: float, strike: float, tau: float, df: float, price: float
) -> float:
    """The call less ``price``: the function ``brentq`` solves."""
    return black76(sigma, fwd, strike, tau, df) - price


def vega(sigma: float, fwd: float, strike: float, tau: float, df: float) -> float:
    v = sigma * np.sqrt(tau)
    d1 = (np.log(fwd / strike) + 0.5 * v * v) / v
    return float(df * fwd * np.sqrt(tau) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi))


def grid(seed: int, n: int) -> list[tuple[float, float, float, float, float, float]]:
    """``n`` quotes: forward ``100 e^U(-1, 1)``, expiries from a day to two
    years, volatilities from 1% to 300%, strikes within three standard
    deviations (``ln(K/F)`` in ``±3 sigma sqrt(tau)``), discount at a rate up
    to 5%; and each one's price at its volatility. Returns ``(F, K, tau, df,
    price, sigma)``."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        fwd = 100.0 * float(np.exp(rng.uniform(-1, 1)))
        tau = float(10 ** rng.uniform(np.log10(1 / 365), np.log10(2)))
        sigma = float(rng.uniform(0.01, 3.0))
        strike = fwd * float(np.exp(3 * sigma * np.sqrt(tau) * rng.uniform(-1, 1)))
        df = float(np.exp(-rng.uniform(0, 0.05) * tau))
        rows.append((fwd, strike, tau, df, black76(sigma, fwd, strike, tau, df), sigma))
    return rows


def line(kind: str, lo: float, hi: float, inc: bool, *p: float) -> str:
    return " ".join(
        [kind, repr(lo), repr(hi), "1" if inc else "0", repr(XTOL), repr(RTOL)]
        + [repr(x) for x in p]
    )


def eps_at(lo: float, hi: float) -> float:
    """The solver's tolerance on ``[lo, hi]``."""
    return XTOL + RTOL * min(abs(lo), abs(hi)) if lo > 0 or hi < 0 else XTOL


@pytest.fixture(scope="module")
def quotes() -> list[tuple[float, float, float, float, float, float]]:
    return grid(563, 4000)


def test_implied_volatility_agrees_with_brentq_wherever_it_is_certified(root_solver, quotes):
    """Measured 2026-11-23 over 4,000 quotes: certified on 3,850, where the
    two roots differ by at most 2.08e-12 (1.96e-10 relative, at a 1%
    volatility), within ``2 eps + xtol + rtol |x|`` (6e-12) on every row;
    the solver took a mean of 18 evaluations (at most 46, its bound at this
    bracket, reached) to ``brentq``'s 12.4, the price of a worst case that is
    stated."""
    out = solve(root_solver, [line("black76", LO, HI, True, *q[:5]) for q in quotes])
    eps = eps_at(LO, HI)
    worst_abs = worst_rel = 0.0
    calls = []
    for (fwd, strike, tau, df, price, sigma), (x, lo, hi, n, bound, why) in zip(
        quotes, out, strict=True
    ):
        assert n <= bound == 46, (n, bound)
        if why != "ok":
            continue
        ref, info = brentq(
            black76_less,
            LO,
            HI,
            args=(fwd, strike, tau, df, price),
            xtol=XTOL,
            rtol=RTOL,
            full_output=True,
        )
        calls.append((n, info.function_calls))
        gap = abs(x - ref)
        assert gap <= 2 * eps + XTOL + RTOL * abs(ref) + 5 * EPS_MACH * abs(x), (x, ref, sigma)
        assert lo <= x <= hi and hi - lo <= 4 * eps + 1e-15
        worst_abs = max(worst_abs, gap)
        worst_rel = max(worst_rel, gap / abs(ref))
    assert len(calls) > 0.95 * len(quotes)
    assert worst_abs < 3e-12 and worst_rel < 3e-10, (worst_abs, worst_rel)
    mine, theirs = np.mean([c[0] for c in calls]), np.mean([c[1] for c in calls])
    print(
        f"certified {len(calls)} of {len(quotes)}: max |d| {worst_abs:.3g}, rel {worst_rel:.3g}; "
        f"evaluations mean {mine:.3g}, brentq {theirs:.3g}"
    )


def test_a_withheld_volatility_is_one_the_price_does_not_fix(root_solver, quotes):
    """Every root withheld from the grid is ``not_determined``, and on each
    the price's rounding moves the volatility by about the tolerance or
    more: ``vega * eps`` is below a few hundred units in the last place of
    the price's larger term ``df max(F, K)`` (measured: 150 withheld, at
    most 220 units among them). On the other side, every quote where it is
    above 1,000 units is certified."""
    out = solve(root_solver, [line("black76", LO, HI, True, *q[:5]) for q in quotes])
    eps = eps_at(LO, HI)
    withheld = 0
    for (fwd, strike, tau, df, _, sigma), (_, _, _, _, _, why) in zip(quotes, out, strict=True):
        units = vega(sigma, fwd, strike, tau, df) * eps / (EPS_MACH * df * max(fwd, strike))
        if why == "ok":
            continue
        withheld += 1
        assert why == "not_determined", why
        assert units < 500, (units, sigma, strike / fwd, tau)
    certain = [
        why
        for (fwd, strike, tau, df, _, sigma), (*_, why) in zip(quotes, out, strict=True)
        if vega(sigma, fwd, strike, tau, df) * eps / (EPS_MACH * df * max(fwd, strike)) > 1000
    ]
    assert certain and all(w == "ok" for w in certain)
    assert 0 < withheld < 0.05 * len(quotes)


def test_a_decreasing_function_gives_the_same_bits_as_its_negation(root_solver, quotes):
    """``price - call``, declared decreasing, against ``call - price``
    declared increasing: every output line the same, to the last digit."""
    rows = quotes[:1000]
    up = solve(root_solver, [line("black76", LO, HI, True, *q[:5]) for q in rows])
    down = solve(root_solver, [line("black76", LO, HI, False, *q[:5]) for q in rows])
    assert [repr(r) for r in up] == [repr(r) for r in down]


def test_a_price_outside_the_functions_range_is_withheld_where_brentq_refuses(root_solver):
    """Below the discounted intrinsic value no volatility gives the price,
    and above the discounted forward none does either: ``brentq`` refuses
    (no sign change) and the solver names the side. At 20% and 250%, both
    solve, and agree."""
    fwd, strike, tau, df = 100.0, 90.0, 0.5, 0.98
    floor, cap = df * (fwd - strike), df * fwd
    inside = (black76(0.2, fwd, strike, tau, df), black76(2.5, fwd, strike, tau, df))
    prices = [floor - 0.01, cap + 0.01, *inside]
    out = solve(
        root_solver, [line("black76", LO, HI, True, fwd, strike, tau, df, p) for p in prices]
    )
    assert [o[5] for o in out[:2]] == ["below_range", "above_range"]
    for p in prices[:2]:
        with pytest.raises(ValueError, match="different signs"):
            brentq(lambda s, p=p: black76(s, fwd, strike, tau, df) - p, LO, HI)
    eps = eps_at(LO, HI)
    for p, (x, *_, why) in zip(prices[2:], out[2:], strict=True):
        assert why == "ok"
        ref = brentq(
            lambda s, p=p: black76(s, fwd, strike, tau, df) - p, LO, HI, xtol=XTOL, rtol=RTOL
        )
        assert abs(x - ref) <= 2 * eps + XTOL + RTOL * abs(ref)


def test_cubics_agree_with_brentq(root_solver):
    """``x^3 + c x - t`` on ``[-4, 4]``, ``c`` in ``[0, 3]``, ``t`` in
    ``[-40, 40]``: a polynomial family, increasing, every root certified
    (its slope is at least ``c`` and grows away from 0), and within the same
    bound of ``brentq``'s; measured at most 2.18e-12 apart, none withheld."""
    rng = np.random.default_rng(3)
    cases = [(float(rng.uniform(0, 3)), float(rng.uniform(-40, 40))) for _ in range(2000)]
    out = solve(root_solver, [line("cubic", -4.0, 4.0, True, c, t) for c, t in cases])
    eps = eps_at(-4.0, 4.0)
    worst = 0.0
    for (c, t), (x, _, _, n, bound, why) in zip(cases, out, strict=True):
        ref = brentq(lambda z, c=c, t=t: z * z * z + c * z - t, -4.0, 4.0, xtol=XTOL, rtol=RTOL)
        if why != "ok":
            # Only a flat cubic near its inflection, c near 0 and t near 0.
            assert why == "not_determined" and c < 1e-3 and abs(ref) < 1e-3, (c, t, why)
            continue
        assert n <= bound
        worst = max(worst, abs(x - ref))
        assert abs(x - ref) <= 2 * eps + XTOL + RTOL * abs(ref), (c, t, x, ref)
    assert worst < 3e-12, worst
