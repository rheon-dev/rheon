"""ABBA timing of two `factor_bench` binaries, with an identical-binary control.

    python3 ab.py --a BIN_A --b BIN_B [--rounds 6] [--reps 5] [filter ...]

Each round runs A, B, A' and B, A', A in turn (A' is a copy of A, so the
control is the identical binary; all three are run from paths of the same
length), every case of the filter,
best-of-`reps` ns a row from each. Per case it prints the median over the
rounds of B/A, its range, and the control A'/A's median and range: a
difference inside the control's range is noise. Digests are checked: A and
A' must agree; B against A is reported (same / moved).
Standard library only, so it runs on any machine with Python 3.
"""

from __future__ import annotations

import argparse
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

LINE = re.compile(r"^(.*?)\s+best\s+([\d.]+) ns/row.*digest ([0-9a-f]+)$")


def run(binary: str, reps: int, filters: list[str]) -> dict[str, tuple[float, str]]:
    out = subprocess.run(
        [binary, str(reps), *filters], check=True, capture_output=True, text=True
    ).stdout
    got = {}
    for line in out.splitlines():
        m = LINE.match(line.strip())
        if m:
            got[m.group(1).strip()] = (float(m.group(2)), m.group(3))
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("filters", nargs="*")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    # All three at paths of the same length: the path is on the stack at
    # start, so its length moves every frame below it, which alone moved an
    # identical binary's time by up to 20% (task 479).
    suffix = Path(args.a).suffix if Path(args.a).suffix == ".exe" else ""
    a1, b1, a2 = (tmp / f"{n}{suffix}" for n in ("binA", "binB", "binC"))
    shutil.copy2(args.a, a1)
    shutil.copy2(args.b, b1)
    shutil.copy2(args.a, a2)
    args.a, args.b = str(a1), str(b1)
    runs: dict[str, list[dict]] = {"A": [], "B": [], "A2": []}
    for r in range(args.rounds):
        order = ["A", "B", "A2"] if r % 2 == 0 else ["B", "A2", "A"]
        for who in order:
            binary = {"A": args.a, "B": args.b, "A2": str(a2)}[who]
            runs[who].append(run(binary, args.reps, args.filters))
        print(f"round {r + 1}/{args.rounds} done", file=sys.stderr, flush=True)
    cases = list(runs["A"][0])
    print(f"{'case':<34} {'A ns':>9} {'B ns':>9} {'B/A med':>8} {'B/A range':>15} "
          f"{'ctl med':>8} {'ctl range':>15}  bits")
    for c in cases:
        a = [x[c][0] for x in runs["A"]]
        b = [x[c][0] for x in runs["B"]]
        a2 = [x[c][0] for x in runs["A2"]]
        ratio = [bi / ai for ai, bi in zip(a, b)]
        ctl = [ci / ai for ai, ci in zip(a, a2)]
        da, db, dc = runs["A"][0][c][1], runs["B"][0][c][1], runs["A2"][0][c][1]
        assert da == dc, f"{c}: the control's digest differs"
        print(
            f"{c:<34} {min(a):>9.1f} {min(b):>9.1f} {statistics.median(ratio):>8.3f} "
            f"{min(ratio):>7.3f}-{max(ratio):<7.3f} {statistics.median(ctl):>8.3f} "
            f"{min(ctl):>7.3f}-{max(ctl):<7.3f}  {'same' if da == db else 'moved'}"
        )
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
