"""ABBA timing harness for Stage 2 (PLAN 583): native binaries and Python wheels,
with an identical-binary control (A, B, A' / B, A', A).

Usage (native binary):
    python3 ab_583.py --mode bin --a BIN_A --b BIN_B [--rounds 6] [--reps 5] [--suite v2] [filter ...]

Usage (Python wheels via benchmark.py / wide_benchmark.py):
    python3 ab_583.py --mode py --a PY_A --b PY_B --script scripts/benchmark.py [--rounds 4] [--extra "--rows 100000"]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

BIN_LINE = re.compile(r"^(.*?)\s+best\s+([\d.]+)\s+ns/row.*digest\s+([0-9a-f]+)$")
PY_MD_LINE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([\d,]+)\s*\|$")
PY_WIDE_LINE = re.compile(r"^(k=\s*\d+\s*\|\s*.+?)\s*\|\s*([\d,]+)\s+rows/s$")


def run_bin(binary: str, reps: int, suite: str, filters: list[str]) -> dict[str, tuple[float, str]]:
    env = dict(os.environ)
    env["RHEON_MAX_THREADS"] = "1"
    env["POLARS_MAX_THREADS"] = "1"
    out = subprocess.run(
        [binary, str(reps), suite, *filters],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout
    got: dict[str, tuple[float, str]] = {}
    for line in out.splitlines():
        m = BIN_LINE.match(line.strip())
        if m:
            got[m.group(1).strip()] = (float(m.group(2)), m.group(3))
    return got


def run_py(python_exe: str, script: str, extra: list[str]) -> dict[str, tuple[float, str]]:
    out = subprocess.run(
        [python_exe, script, *extra],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    got: dict[str, tuple[float, str]] = {}
    for line in out.splitlines():
        s = line.strip()
        m = PY_WIDE_LINE.match(s)
        if m:
            key = re.sub(r"\s+", " ", m.group(1).strip())
            rps = float(m.group(2).replace(",", ""))
            # Convert rows/s to ns/row so B/A > 1 means slower, consistent with native mode.
            got[key] = (1e9 / rps, "-")
            continue
        m = PY_MD_LINE.match(s)
        if m:
            key = f"{m.group(1).strip()} ({m.group(2).strip()})"
            rps = float(m.group(3).replace(",", ""))
            got[key] = (1e9 / rps, "-")
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bin", "py"], required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--suite", default="v2")
    ap.add_argument("--script", default="")
    ap.add_argument("--extra", default="")
    ap.add_argument("filters", nargs="*")
    args = ap.parse_args()

    tmp: Path | None = None
    if args.mode == "bin":
        tmp = Path(tempfile.mkdtemp())
        suffix = Path(args.a).suffix if Path(args.a).suffix == ".exe" else ""
        a1, b1, a2 = (tmp / f"{n}{suffix}" for n in ("binA", "binB", "binC"))
        shutil.copy2(args.a, a1)
        shutil.copy2(args.b, b1)
        shutil.copy2(args.a, a2)
        targets = {"A": str(a1), "B": str(b1), "A2": str(a2)}
    else:
        targets = {"A": args.a, "B": args.b, "A2": args.a}

    extra = args.extra.split() if args.extra else []
    runs: dict[str, list[dict[str, tuple[float, str]]]] = {"A": [], "B": [], "A2": []}
    for r in range(args.rounds):
        order = ["A", "B", "A2"] if r % 2 == 0 else ["B", "A2", "A"]
        for who in order:
            if args.mode == "bin":
                res = run_bin(targets[who], args.reps, args.suite, args.filters)
            else:
                res = run_py(targets[who], args.script, extra)
            runs[who].append(res)
        print(f"round {r + 1}/{args.rounds} done", file=sys.stderr, flush=True)

    # Intersect keys present in both A and B (e.g. benchmark.py added post-2.0.0 rows on HEAD)
    cases = [c for c in runs["A"][0] if c in runs["B"][0]]
    print(
        f"{'case':<48} {'A ns/r':>9} {'B ns/r':>9} {'B/A med':>8} {'B/A range':>15} "
        f"{'ctl med':>8} {'ctl range':>15}  flag"
    )
    for c in cases:
        a = [x[c][0] for x in runs["A"]]
        b = [x[c][0] for x in runs["B"]]
        a2 = [x[c][0] for x in runs["A2"]]
        ratio = [bi / ai for ai, bi in zip(a, b, strict=True)]
        ctl = [ci / ai for ai, ci in zip(a, a2, strict=True)]
        med_r = statistics.median(ratio)
        ctl_lo, ctl_hi = min(ctl), max(ctl)
        da, db, dc = runs["A"][0][c][1], runs["B"][0][c][1], runs["A2"][0][c][1]
        if args.mode == "bin":
            assert da == dc, f"{c}: identical-binary control digest differed"
        outside = "SLOWER" if med_r > max(ctl_hi, 1.03) else ("FASTER" if med_r < min(ctl_lo, 0.97) else "level")
        bits = "" if da == "-" else (" same" if da == db else " moved")
        print(
            f"{c:<48} {min(a):>9.1f} {min(b):>9.1f} {med_r:>8.3f} "
            f"{min(ratio):>7.3f}-{max(ratio):<7.3f} {statistics.median(ctl):>8.3f} "
            f"{ctl_lo:>7.3f}-{ctl_hi:<7.3f}  {outside}{bits}"
        )
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
