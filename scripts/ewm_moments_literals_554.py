"""Print the reference numbers `crates/rheon-core/src/ewm_moments/tests.rs`
holds the Rust state to (PLAN 554, part 3).

The stream is the one the Rust test generates: the integer generator of
`tests/model_contract.rs` and arithmetic alone, so both sides form the same
doubles. The reference (`tests/reference_ewm_moments.py`) is run on it with
the output read after each row (`emit="after"`), and the numbers at a few
rows are printed as Rust literals, each with its bits.

    python3 scripts/ewm_moments_literals_554.py

Needs nothing but the standard library: the reference is pure Python.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tests"))

import reference_ewm_moments as rm  # noqa: E402

N = 60
CHECK_ROWS = (2, 31, 59)


def lcg(state: int) -> tuple[int, float]:
    state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
    return state, (state >> 11) / 2**53


def stream(seed: int, zero_rows: tuple[int, ...], steps: bool) -> dict[str, list[float]]:
    """`x = 3 + 4u^3 + v/2` (skewed), `w = 1/4 + u'`, a weight of 0 on
    `zero_rows`, and a clock stepping 1, 1.5, 2, ... (sums of halves, exact)
    under `steps`, else 1 a row."""
    s = seed
    x, w, t = [], [], []
    clock = 0.0
    for i in range(N):
        s, u = lcg(s)
        s, v = lcg(s)
        s, q = lcg(s)
        x.append(3.0 + 4.0 * u * u * u + 0.5 * v)
        w.append(0.0 if i in zero_rows else 0.25 + q)
        clock += 1.0 + (i % 3) * 0.5 if steps else 1.0
        t.append(clock)
    return {"x": x, "w": w, "t": t}


def terms(bias: bool) -> tuple[dict, dict]:
    top = 6 if bias else 4
    tm: dict[str, dict[str, int]] = {}
    kd: dict[str, str] = {}
    for p in range(2, top + 1):
        tm[f"c{p}"] = {"x": p}
        kd[f"c{p}"] = "central"
        tm[f"k{p}"] = {"x": p}
        kd[f"k{p}"] = "cumulant"
    for p in range(3, top + 1):
        tm[f"s{p}"] = {"x": p}
        kd[f"s{p}"] = "standardized"
    if bias:
        for p in range(1, 7):
            tm[f"r{p}"] = {"x": p}
            kd[f"r{p}"] = "raw"
    return tm, kd


def run(data, bias: bool, **decay) -> None:
    tm, kd = terms(bias)
    spec = rm.EwmMoments(tm, kind=kd, bias=bias, clocked=True, emit="after", **decay)
    for i in range(N):
        part = {k: [v[i]] for k, v in data.items()}
        out = spec.feed(part)
        if i in CHECK_ROWS:
            (summ,) = spec.sums.values()
            rows = [("W", summ.W)] if bias else []
            rows += [("kish_r2", summ.r[0])] if bias else []
            rows += [(k + ("" if bias else "u"), o[0].value) for k, o in out.items()]
            for k, v in rows:
                lit = "f64::NAN" if v is None else repr(v)
                print(f'    ({i}, "{k}", {lit}),')


def main() -> None:
    # A: a zero-weight first row and no other, one clock unit a row under
    # lam = 0.9, so the weight and Kish's ratio go through no libm and are
    # the Rust's to the bit on every platform.
    a = stream(554, (0,), False)
    print("const REF_A: &[(usize, &str, f64)] = &[")
    run(a, True, lam=0.9)
    run(a, False, lam=0.9)
    print("];")
    # B: zero-weight rows inside, a half-life of 7 clock units.
    b = stream(5541, (0, 17, 18, 40), True)
    print("const REF_B: &[(usize, &str, f64)] = &[")
    run(b, True, half_life=7.0)
    run(b, False, half_life=7.0)
    print("];")


if __name__ == "__main__":
    main()
