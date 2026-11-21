"""Study 501 and 509: expectiles and HyperLogLog, every check, every table.

    python scripts/study_501_509.py            # every table (about 3 minutes)
    python scripts/study_501_509.py --quick    # fewer seeds and trials (about 1 minute)
    python scripts/study_501_509.py --only hll # one half: expectile or hll

Any Python with numpy, scipy, statsmodels and datasketch runs it (datasketch
is not in the dev group: a venv of its own, ``pip install numpy scipy
statsmodels datasketch``). The prototypes and the checks live in
``studies/study_501_509/``; the write-up is ``docs/records/STUDY-501-509.md``.
Nothing here draws a seed at run time: every stream is seeded by its index.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "study_501_509"))
import compare_expectile as CE  # noqa: E402
import compare_hll as CH  # noqa: E402
import datasketch  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import statsmodels  # noqa: E402


def cell(c: object) -> str:
    if c is None:
        return "-"
    if isinstance(c, (bool, np.bool_)):
        return "yes" if c else "no"
    if isinstance(c, (float, np.floating)):
        c = float(c)
        if c != c:
            return "nan"
        if c == 0.0:
            return "0"
        if c == int(c) and abs(c) < 1e7:
            return str(int(c))
        if abs(c) < 1e-3 or abs(c) >= 1e6:
            return f"{c:.2e}"
        return f"{c:.4g}"
    return str(c)


def table(title: str, hdr, rows) -> None:
    print(f"\n### {title}\n")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(cell(c) for c in r) + " |")
    sys.stdout.flush()


def pairs(title: str, rows) -> None:
    table(title, ("item", "value"), rows)


def expectile_part(q: bool) -> None:
    s = 3 if q else 20
    table("E1. Scalar expectile: exact sort against scipy.stats.expectile",
          *CE.scalar_vs_scipy(seeds=s))
    table("E2. Scalar expectile, one pass, against scipy on every row (decayed)",
          *CE.scalar_online_vs_scipy(seeds=s))
    pairs("E3. tau = 1/2: one pass against the decayed WLS (max rel gap)",
          [("max |beta - beta_WLS| / max |beta_WLS|", CE.tau_half_is_wls(seeds=s))])
    table("R1. Exact refit against the WLS longhand's stationarity, statsmodels WLS iterated, "
          "statsmodels RLM(MQuantileNorm)", *CE.refit_vs_oracles(seeds=2 if q else 10))
    total, flips, zeros = CE.one_pass_inner()
    pairs("R2. One pass: the row's side judged before and after its own update",
          [("rows", total), ("sides that differ", flips), ("residuals after at exactly 0", zeros)])
    table("R3. One pass (and with the first 200 rows refit) against the exact decayed fit",
          *CE.online_vs_batch(seeds=2 if q else 10))
    table("R4. Window: exact refit each row against statsmodels on the window's rows",
          *CE.window_vs_batch(seeds=1 if q else 4))
    table("R5. Passes per refit: warm start against cold start (window 500, k = 5)",
          *CE.warm_vs_cold(seeds=1 if q else 5))
    table("R6. Convergence on awkward problems (cold start)",
          *CE.convergence_stress(cases=300 if q else 1_000))
    table("R7. Zero-weight rows (rule 9)", *CE.zero_weight())


def hll_part(q: bool) -> None:
    table("H1. Against datasketch with this hash handed in (hashfunc=)",
          *CH.vs_datasketch(n=10_000 if q else 50_000))
    table("H2. datasketch's default hash (SHA-1 of the key's 8 bytes)", *CH.default_hash())
    table("H3. Avalanche: one input bit flipped", *CH.avalanche(n=1_024 if q else 4_096))
    t = 40 if q else 200
    table(f"H4. Error band at n = 20 m, random values, {t} trials a precision",
          *CH.error_band(trials=t))
    table(f"H4b. The same on sequential integers, {t} trials a precision",
          *CH.error_band(trials=t, kind="sequential"))
    table("H5. Across n at p = 10: the linear-counting hand-over",
          *CH.small_range(p=10, trials=t))
    table("H6. Hash quality on structured values (p = 12, n = 20 m)",
          *CH.hash_quality(p=12, trials=20 if q else 100))
    table("H7. Merge and order", *CH.merge_checks())
    pairs("H8. Keys", CH.keying())
    pairs("H9. Zero-weight rows", CH.zero_weight())
    digests, pins = CH.determinism()
    pairs("H10. Determinism", [("register digests over 4 PYTHONHASHSEEDs", ", ".join(digests))]
          + [(a, f"{b} ({'ok' if c else 'MISMATCH'})") for a, b, c in pins])
    table("H11. Bytes against precision", *CH.sizes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=("expectile", "hll"))
    a = ap.parse_args()
    print(f"Python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, "
          f"statsmodels {statsmodels.__version__}, datasketch {datasketch.__version__}, "
          f"{platform.system()} {platform.machine()}")
    t = time.time()
    if a.only in (None, "expectile"):
        expectile_part(a.quick)
    if a.only in (None, "hll"):
        hll_part(a.quick)
    print(f"\n({time.time() - t:.0f} s)")


if __name__ == "__main__":
    main()
