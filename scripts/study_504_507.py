"""Study 504/507: a counter-based draw stream, contextual bandits on ridge states,
and streaming anomaly detectors that draw from it; every table.

    python scripts/study_504_507.py            # every table (about 6 minutes)
    python scripts/study_504_507.py --quick    # fewer rows and seeds (about 1 minute)

Needs numpy, scipy, scikit-learn, mpmath, river, mabwiser, rrcf (whose import
needs setuptools < 81 for ``pkg_resources``) and vowpalwabbit. The code lives in
``studies/study_504_507/``; the write-up is ``docs/records/STUDY-504-507.md``.
Every stream is simulated from a seed; nothing is read from disk.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "study_504_507"))
warnings.filterwarnings("ignore")

import bandits as B  # noqa: E402
import hst as H  # noqa: E402
import numpy as np  # noqa: E402
import rcf as R  # noqa: E402
import streamcheck as S  # noqa: E402


def cell(c: object) -> str:
    if c is None:
        return "-"
    if isinstance(c, (bool, np.bool_)):
        return "yes" if c else "no"
    if isinstance(c, (int, np.integer)):
        return f"{int(c):,}"
    if isinstance(c, (float, np.floating)):
        c = float(c)
        if c == 0:
            return "0"
        if abs(c) < 1e-3 or abs(c) >= 1e6:
            return f"{c:.2e}"
        return f"{c:.4g}"
    return str(c)


def table(title: str, head: list[str], rows: list) -> None:
    print(f"\n### {title}\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(cell(c) for c in r) + " |")
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    q = a.quick
    t0 = time.time()
    libs = ["numpy", "scipy", "scikit-learn", "mpmath", "river", "mabwiser", "rrcf", "vowpalwabbit"]
    print("# Study 504/507: every table\n")
    print(
        f"Python {platform.python_version()} on {platform.platform()}; "
        + ", ".join(f"{n} {version(n)}" for n in libs)
    )

    # ---- (1) the generator ------------------------------------------------------------
    table("G1. Known answers (Random123's kat_vectors)", ["check", "equal", "of"], S.check_kats())
    table(
        "G2. The longhand Philox4x64-10 against numpy.random.Philox",
        ["check", "equal", "of"],
        S.check_numpy(1000 if q else 3000),
    )
    nm = S.check_normals(5000 if q else 20000)
    table(
        "G3. Normals: Wichura's AS 241 (PPND16) on (0,1) uniforms",
        ["measure", "value"],
        list(nm.items()),
    )
    table(
        f"G4. A drawing consumer over {2000 if q else 5000:,} rows, 40 groups, "
        "20% zero weights (first row zero): "
        "digest of every "
        "emitted draw",
        ["chunking", "digest"],
        S.check_invariance(2000 if q else 5000),
    )
    table(
        "G5. Two processes per salt: the stream's digest, and a digest keyed on Python's hash(str)",
        ["process", "stream (FixedHasher)", "keyed on hash()"],
        S.check_processes(),
    )
    table("G6. Zero-weight rows and the draw position", ["check", "holds"], S.check_zero_weight())

    # ---- (2) bandits --------------------------------------------------------------------
    T = 400 if q else 1500
    rows, cov_err, ties = B.oracle_table(T=T)
    table(
        f"B1. Same draws, same logged actions, {T} rows, 5 arms, 6 features (driver's choice is "
        f"learned)",
        [
            "policy (driver)",
            "against",
            "rows choosing differently",
            "max |score diff|",
            "max relative",
        ],
        rows,
    )
    table(
        "B2. Checks on the draws' transforms",
        ["check", "value"],
        [
            (
                "numpy's multivariate_normal('cholesky') = mean + z @ chol(cov).T, to the bit "
                "(20 cases)",
                B.mvn_transform_check(),
            ),
            (
                "fast root: max rel. error of (aR^-1)(aR^-1)' vs a^2 A^-1 over the arms, end of B1",
                cov_err,
            ),
            ("rows where river's rule broke a tie (lane-0 draws used)", ties),
        ],
    )
    table(
        "B3. Chunking and save/resume, logged stream with zero-weight rows (first two zero)",
        ["policy", "runs", "distinct digests", "digest"],
        B.bandit_invariance(500 if q else 1500),
    )
    table("B4. Zero-weight rows", ["check", "holds"], B.zero_weight_checks())
    TR, SE = (600, 3) if q else (2000, 8)
    table(
        f"B5. Cumulative pseudo-regret, {TR} rows, 5 arms, 6 features, noise sd 0.5, mean over "
        f"{SE} seeds",
        ["policy", f"at {TR // 2}", f"at {TR}", "sd at the end"],
        B.regret_table(T=TR, seeds=SE),
    )
    table(
        f"B6. Arms change at row {TR // 2}: regret over the second half (rheon form, decay on "
        f"every learned row)",
        ["policy", "lam", "regret, 2nd half", "sd"],
        B.drift_table(T=TR, seeds=SE),
    )
    table(
        "B7. Bandit state: K * 8 * (d(d+1)/2 + d) + 8 bytes (R's triangle and u per arm, the "
        "learned-row count)",
        ["arms K", "features d", "bytes"],
        B.state_bytes_table(),
    )

    # ---- (3) half-space trees -------------------------------------------------------------
    cfgs = [
        (3, 10, 8, 250, None),
        (2, 25, 6, 100, None),
        (5, 5, 10, 50, None),
        (4, 10, 4, 64, [(-3.0, 3.0), (0.0, 1.0), (10.0, 20.0), (0.0, 1e4)]),
    ]
    rows = []
    for d, nt, h, w, lim in cfgs:
        for src in ["river", "philox"]:
            r = H.compare(600 if q else 2000, d, nt, h, w, seed=d + nt, limits=lim, source=src)
            rows.append(
                (
                    f"d={d}, {nt} trees, h={h}, window {w}" + (", limits" if lim else ""),
                    src,
                    r["draws"],
                    r["features equal"],
                    r["splits equal (bits)"],
                    r["masses equal at the end"],
                    f"{r['scores equal to the bit']}/{r['rows']}",
                    r["max |score diff|"],
                    r["recording changes nothing"],
                )
            )
    table(
        "H1. Half-space trees: the port against river.anomaly.HalfSpaceTrees, same construction "
        "draws",
        [
            "config",
            "draws",
            "uniforms",
            "features",
            "splits (bits)",
            "masses (end)",
            "scores to the bit",
            "max |diff|",
            "recording = plain seed",
        ],
        rows,
    )
    table(
        "H2. Chunking and save/resume (trees rebuilt from the seed at load), weights 0/0.5/1/2, "
        "first 3 zero",
        ["runs", "distinct digests", "digest", "zero-weight row leaves the state"],
        [H.invariance(1000 if q else 3000)],
    )
    table(
        "H3. Half-space tree state per group: masses 16 B a node; + 12 B an internal node if the "
        "trees are kept",
        ["trees", "height", "nodes", "bytes (trees rebuilt)", "bytes (trees kept)"],
        H.state_table(),
    )

    # ---- (4) random cut forest -------------------------------------------------------------
    rows = []
    w = np.random.default_rng(1).choice([0.0, 1.0, 1.0, 1.0], 600)
    w[0] = 0.0
    for name, kw in [
        ("stream's draws, shingle 4, 8 trees of 64", {}),
        ("rrcf's own MT19937 draws (RandomState per tree)", {"source": "mt"}),
        ("stream's draws, 25% zero weights, first zero", {"weights": w}),
        (
            "stream's draws, shingle 10, 4 trees of 32",
            {"shingle": 10, "num_trees": 4, "tree_size": 32},
        ),
    ]:
        r = R.run(T=600, **kw)
        rows.append(
            (
                name,
                f"{r['scores equal to the bit']}/{r['rows']}",
                r["max |score diff|"],
                r["trees equal at the end (cut, value bits, count)"],
                r["recording changes nothing"],
                r["learned rows"],
            )
        )
    table(
        "R1. The port in rrcf's arithmetic against rrcf 0.4.4, same draws, 600 rows",
        [
            "case",
            "scores to the bit",
            "max |diff|",
            "trees equal (end)",
            "recording = plain seed",
            "learned rows",
        ],
        rows,
    )
    mt = R.modes_table(T=800 if q else 1500, num_trees=10 if q else 20, tree_size=64 if q else 128)
    table(
        "R2. rrcf's leaf cut against the paper's, same draws (AUC: score against the injected "
        "anomaly rows)",
        list(mt[0].keys()),
        [list(m.values()) for m in mt],
    )
    (rb, rt), (pb, pt) = R.violations_table(T=400 if q else 800)
    table(
        "R3. Branches whose cut does not separate its children's points (the paper's invariant)",
        ["tree", "violating", "branches"],
        [("rrcf 0.4.4", rb, rt), ("port, paper's cut", pb, pt)],
    )
    table(
        "R4. Paper-mode forest: chunking and a pickle round trip after every chunk",
        ["runs", "distinct digests", "digest"],
        [R.invariance(300 if q else 800)],
    )
    table("R4b. Zero-weight rows in the forest", ["check", "holds"], R.zero_weight_check())
    table(
        "R5. Random cut forest state per group (no cap built in rrcf; tree_size is the window)",
        ["trees", "tree size", "d", "bytes"],
        R.state_table(),
    )

    print(f"\n(all tables in {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
